import { Anthropic } from "@anthropic-ai/sdk"
import OpenAI from "openai"
import { ApiHandler } from "../"
import { ApiHandlerOptions, ModelInfo, openAiModelInfoSaneDefaults } from "../../shared/api"
import { convertToOpenAiMessages } from "../transform/openai-format"
import { ApiStream } from "../transform/stream"
import { Transform, TransformCallback } from "node:stream"

import "whatwg-fetch"

import { withRetry } from "../retry"

interface ChatRequest {
	llmId: number
	contents: string[]
	isStream: boolean
}

interface ChatResponse {
	id: number
	model_type: string
	content: string
	completion_token: number
	prompt_token: number
	finish_reason: string
	filter_block_reason: {
		ko: string
		en: string
		policy_id: number
		message: string
		result_code: string
		filter_log_id: number
	}
	status: string
	event_status: string
	event_data: string
}

interface ErrorResponse {
	error: string
}

class AbortableAsyncIterator<T extends object> {
	private readonly abortController: AbortController
	private readonly itr: AsyncGenerator<T | ErrorResponse>
	private readonly doneCallback: () => void

	constructor(abortController: AbortController, itr: AsyncGenerator<T | ErrorResponse>, doneCallback: () => void) {
		this.abortController = abortController
		this.itr = itr
		this.doneCallback = doneCallback
	}

	abort() {
		this.abortController.abort()
	}

	async *[Symbol.asyncIterator]() {
		for await (const message of this.itr) {
			if ("error" in message) {
				throw new Error(message.error)
			}
			yield message
			if ((message as any).done || (message as any).status === "success") {
				this.doneCallback()
				return
			}
		}
		throw new Error("Did not receive done or success response in stream.")
	}
}

const parseJSON = async function* <T = unknown>(itr: ReadableStream<Uint8Array>): AsyncGenerator<T> {
	const decoder = new TextDecoder("utf-8")
	let buffer = ""
	const reader = itr.getReader()
	while (true) {
		const { done, value: chunk } = await reader.read()
		if (done) {
			break
		}
		buffer += decoder.decode(chunk)
		const parts = buffer.split("\n")
		buffer = parts.pop() ?? ""

		for (const part of parts) {
			try {
				yield JSON.parse(part)
			} catch (error) {
				console.warn("invalid json: ", part)
			}
		}
	}

	for (const part of buffer.split("\n").filter((p) => p !== "")) {
		try {
			yield JSON.parse(part)
		} catch (error) {
			console.warn("invalid json: ", part)
		}
	}
}

export class FabrixHandler implements ApiHandler {
	private options: ApiHandlerOptions
	protected readonly ongoingStreamedRequests: AbortableAsyncIterator<object>[] = []

	constructor(options: ApiHandlerOptions) {
		this.options = options
	}

	async chat(systemPrompt: string, messages: Anthropic.Messages.MessageParam[]): Promise<AbortableAsyncIterator<ChatResponse>> {
		return this.processStreamableRequest<ChatResponse>(systemPrompt, messages)
	}

	async processStreamableRequest<T extends object>(
		systemPrompt: string,
		messages: Anthropic.Messages.MessageParam[],
	): Promise<AbortableAsyncIterator<T>> {
		const abortController = new AbortController()
		const url = `${this.options.fabrixBaseUrl}/openapi/chat/v1/messages`
		const body: ChatRequest = {
			llmId: Number(this.options.fabrixModelId),
			contents: messages.map((message) => String(message.content)),
			isStream: true,
		}
		const response = await fetch(url, {
			method: "POST",
			body: JSON.stringify(body),
			headers: {
				"x-generative-ai-client": String(this.options.fabrixToken),
			},
			signal: abortController.signal,
		})

		if (!response.body) {
			throw new Error("Missing body")
		}

		const itr = parseJSON<T | ErrorResponse>(response.body)
		const abortableAsyncIterator = new AbortableAsyncIterator(abortController, itr, () => {
			const i = this.ongoingStreamedRequests.indexOf(abortableAsyncIterator)
			if (i > -1) {
				this.ongoingStreamedRequests.splice(i, 1)
			}
		})
		this.ongoingStreamedRequests.push(abortableAsyncIterator)
		return abortableAsyncIterator
	}

	// async *createMessage(systemPrompt: string, messages: Anthropic.Messages.MessageParam[]): ApiStream {
	// 	const stream = await this.chat(systemPrompt, messages)
	// 	for await (const chunk of stream) {
	// 		if (typeof chunk.content === "string") {
	// 			yield {
	// 				type: "text",
	// 				text: chunk.content,
	// 			}
	// 		}
	// 	}
	// }

	@withRetry({ retryAllErrors: true })
	async *createMessage(systemPrompt: string, messages: Anthropic.Messages.MessageParam[]): ApiStream {
		try {
			// Create a promise that rejects after timeout
			const timeoutPromise = new Promise<never>((_, reject) => {
				setTimeout(() => reject(new Error("Ollama request timed out after 30 seconds")), 30000)
			})

			// Create the actual API request promise
			const apiPromise = this.chat(systemPrompt, messages)

			// Race the API request against the timeout
			const stream = (await Promise.race([apiPromise, timeoutPromise])) as Awaited<typeof apiPromise>

			try {
				for await (const chunk of stream) {
					if (typeof chunk.content === "string") {
						yield {
							type: "text",
							text: chunk.content,
						}

						yield {
							type: "usage",
							inputTokens: chunk.prompt_token || 0,
							outputTokens: chunk.completion_token || 0,
						}
					}
				}
			} catch (streamError: any) {
				console.error("Error processing Fabrix stream:", streamError)
				throw new Error(`Fabrix stream processing error: ${streamError.message || "Unknown error"}`)
			}
		} catch (error: any) {
			// Check if it's a timeout error
			if (error.message && error.message.includes("timed out")) {
				throw new Error("Fabrix request timed out after 30 seconds")
			}

			// Enhance error reporting
			const statusCode = error.status || error.statusCode
			const errorMessage = error.message || "Unknown error"

			console.error(`Fabrix API error (${statusCode || "unknown"}): ${errorMessage}`)
			throw error
		}
	}

	getModel(): { id: string; info: ModelInfo } {
		return {
			id: this.options.fabrixModelId || "",
			info: openAiModelInfoSaneDefaults,
		}
	}
}
