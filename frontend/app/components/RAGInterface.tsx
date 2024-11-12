'use client';

import React, { useState, useEffect, useCallback, useRef, memo } from 'react';
import { Upload, File as FileIcon, CheckCircle, X, Loader2, Send } from 'lucide-react';
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import dynamic from 'next/dynamic';
import { v4 as uuidv4 } from 'uuid';

// Dynamically import ReactMarkdown with no SSR
const ReactMarkdown = dynamic(() => import('react-markdown'), { ssr: false });

// Backend URL moved outside the component
const backendUrl = process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:8000';

// Type definitions
interface MarkdownDisplayProps {
  content: string;
}

interface FileDropZoneProps {
  onFileSelect: (files: File[]) => void;
  unprocessedFiles: File[];
  isUploading: boolean;
}

interface Message {
  id: string; // Unique ID for each message
  role: 'user' | 'assistant';
  content: string;
  timestamp?: string;
}

// MarkdownDisplay Component
const MarkdownDisplay: React.FC<MarkdownDisplayProps> = ({ content }) => {
    const [isClient, setIsClient] = useState(false);

    useEffect(() => {
        setIsClient(true);
    }, []);

    if (!isClient) return null;

    return (
        <ReactMarkdown 
            className="prose prose-blue max-w-none"
            components={{
                h1: (props) => <h1 className="text-2xl font-bold my-4" {...props} />,
                h2: (props) => <h2 className="text-xl font-bold my-3" {...props} />,
                h3: (props) => <h3 className="text-lg font-bold my-2" {...props} />,
                ul: (props) => <ul className="list-disc ml-4 my-2 space-y-2" {...props} />,
                ol: (props) => <ol className="list-decimal ml-4 my-2 space-y-2" {...props} />,
                li: (props) => <li className="my-1" {...props} />,
                p: (props) => <p className="my-2 leading-relaxed" {...props} />,
                strong: (props) => <strong className="font-bold text-gray-900" {...props} />,
                em: (props) => <em className="italic text-gray-800" {...props} />
            }}
        >
            {content}
        </ReactMarkdown>
    );
};

// Base ChatMessage Component with forwardRef
const ChatMessageComponent = React.forwardRef<HTMLDivElement, { message: Message }>(({ message }, ref) => (
  <div
    ref={ref} // Assign ref to both user and assistant messages
    className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'} mb-4`}
  >
    {message.role === 'assistant' && (
      <Avatar className="mr-2">
        <AvatarFallback>AI</AvatarFallback>
      </Avatar>
    )}
    <div className={`rounded-lg p-4 max-w-sm ${
      message.role === 'user' ? 'bg-blue-500 text-white' : 'bg-gray-200 text-gray-800'
    }`}>
      {message.role === 'assistant' ? (
        <MarkdownDisplay content={message.content} />
      ) : (
        <span>{message.content}</span>
      )}
      <div className="text-xs mt-1 opacity-70">
        {message.timestamp ? message.timestamp : '—'}
      </div>
    </div>
    {message.role === 'user' && (
      <Avatar className="ml-2">
        <AvatarFallback>U</AvatarFallback>
      </Avatar>
    )}
  </div>
));

// Assign displayName before memoizing
ChatMessageComponent.displayName = 'ChatMessage';

// Memoize the ChatMessage component
const ChatMessage = memo(ChatMessageComponent);

// Main Component
const RAGInterface: React.FC = () => {
    const chatContainerRef = useRef<HTMLDivElement>(null);
    const messagesEndRef = useRef<HTMLDivElement>(null);
    const inputRef = useRef<HTMLInputElement>(null);

    const [unprocessedFiles, setUnprocessedFiles] = useState<File[]>([]);
    const [processedFiles, setProcessedFiles] = useState<File[]>([]);
    const [messages, setMessages] = useState<Message[]>([]);
    const [input, setInput] = useState('');
    const [uploadLoading, setUploadLoading] = useState(false);
    const [questionLoading, setQuestionLoading] = useState(false);
    const [error, setError] = useState('');
    const [uploadSuccess, setUploadSuccess] = useState(false);
    const [processingFiles, setProcessingFiles] = useState<{[key: string]: string}>({});
    const [sessionId] = useState(() => uuidv4());
    const [shouldAutoScroll, setShouldAutoScroll] = useState(true);

    // Scroll to bottom using scrollIntoView
    const scrollToBottom = useCallback(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, []);

    // Auto-scroll when messages change
    useEffect(() => {
        if (shouldAutoScroll) {
            scrollToBottom();
        }
    }, [messages, scrollToBottom, shouldAutoScroll]);

    // Handle user scroll to toggle auto-scroll
    useEffect(() => {
        const chatContainer = chatContainerRef.current;
        if (!chatContainer) return;

        const handleScroll = () => {
            const { scrollTop, scrollHeight, clientHeight } = chatContainer;
            const isAtBottom = Math.abs(scrollHeight - scrollTop - clientHeight) < 5;
            setShouldAutoScroll(isAtBottom);
        };

        chatContainer.addEventListener('scroll', handleScroll);
        return () => chatContainer.removeEventListener('scroll', handleScroll);
    }, []);

    // Effect to manage body scroll when input is focused on mobile
    useEffect(() => {
        const handleFocus = () => {
            if (window.innerWidth <= 768) {
                document.body.style.overflow = 'hidden';
            }
        };
        const handleBlur = () => {
            if (window.innerWidth <= 768) {
                document.body.style.overflow = 'auto';
            }
        };

        const inputElement = inputRef.current;
        if (inputElement) {
            inputElement.addEventListener('focus', handleFocus);
            inputElement.addEventListener('blur', handleBlur);
        }

        return () => {
            if (inputElement) {
                inputElement.removeEventListener('focus', handleFocus);
                inputElement.removeEventListener('blur', handleBlur);
            }
            document.body.style.overflow = 'auto';
        };
    }, []);

    // File Drop Zone Component
    const FileDropZone: React.FC<FileDropZoneProps> = ({ onFileSelect, unprocessedFiles, isUploading }) => {
        const [isDragging, setIsDragging] = useState(false);

        const handleDrag = useCallback((e: React.DragEvent) => {
            e.preventDefault();
            e.stopPropagation();
        }, []);

        const handleDragIn = useCallback((e: React.DragEvent) => {
            e.preventDefault();
            e.stopPropagation();
            setIsDragging(true);
        }, []);

        const handleDragOut = useCallback((e: React.DragEvent) => {
            e.preventDefault();
            e.stopPropagation();
            setIsDragging(false);
        }, []);

        const handleDrop = useCallback((e: React.DragEvent) => {
            e.preventDefault();
            e.stopPropagation();
            setIsDragging(false);

            const droppedFiles = Array.from(e.dataTransfer.files)
                .filter(file => file.type === 'application/pdf')
                .slice(0, 10 - unprocessedFiles.length);

            if (droppedFiles.length > 0) {
                onFileSelect(droppedFiles);
            }
        }, [onFileSelect, unprocessedFiles]);

        return (
            <div
                onDragEnter={handleDragIn}
                onDragLeave={handleDragOut}
                onDragOver={handleDrag}
                onDrop={handleDrop}
                className={`relative border-2 border-dashed rounded-xl p-8 text-center transition-all duration-200 
                    ${isDragging ? 'border-blue-500 bg-blue-50' : 'border-gray-300 hover:border-blue-400'}`}
            >
                <input
                    type="file"
                    accept=".pdf"
                    multiple
                    onChange={(e: React.ChangeEvent<HTMLInputElement>) => {
                        if (e.target.files) {
                            const newFiles = Array.from(e.target.files).slice(0, 10 - unprocessedFiles.length);
                            onFileSelect(newFiles);
                        }
                    }}
                    className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
                    aria-label="File Upload Input"
                />

                {/* Displaying Unprocessed Files */}
                <div className="space-y-4">
                    {unprocessedFiles.length > 0 ? (
                        <div className="space-y-2">
                            {unprocessedFiles.map((file, index) => (
                                <div key={index} className="flex items-center justify-center gap-3">
                                    <FileIcon className="w-5 h-5 text-blue-600" aria-hidden="true" />
                                    <span className="text-gray-700 font-medium">{file.name}</span>
                                    <span className="text-gray-500 text-sm">({(file.size / 1024).toFixed(2)} KB)</span>
                                </div>
                            ))}
                        </div>
                    ) : (
                        <>
                            <Upload className="w-12 h-12 text-blue-600 mx-auto" aria-hidden="true" />
                            <div className="space-y-2">
                                <p className="text-gray-700 font-medium">
                                    Drag & drop your PDFs here or click to browse
                                </p>
                                <p className="text-sm text-gray-500">
                                    (Maximum 10 files, 10MB each)
                                </p>
                            </div>
                        </>
                    )}
                </div>

                {isUploading && (
                    <div className="absolute inset-0 bg-white bg-opacity-90 flex items-center justify-center">
                        <Loader2 className="w-8 h-8 text-blue-600 animate-spin" aria-label="Uploading" />
                    </div>
                )}
            </div>
        );
    };

    // Handle File Selection with File Size Validation
    const handleFileSelect = useCallback((selectedFiles: File[]) => {
        const maxFileSizeMB = 10;
        const validFiles: File[] = [];
        const oversizedFiles: string[] = [];

        selectedFiles.forEach(file => {
            if (file.size <= maxFileSizeMB * 1024 * 1024) {
                validFiles.push(file);
            } else {
                oversizedFiles.push(file.name);
            }
        });

        if (oversizedFiles.length > 0) {
            setError(`The following files exceed ${maxFileSizeMB}MB and were not added: ${oversizedFiles.join(', ')}`);
        }

        setUnprocessedFiles(prev => [...prev, ...validFiles].slice(0, 10));
    }, []);

    // Handle File Upload
    const handleUpload = useCallback(async () => {
        if (unprocessedFiles.length === 0) {
            setError('Please select at least one file');
            return;
        }

        setUploadLoading(true);
        setError('');

        try {
            for (const file of unprocessedFiles) {
                const formData = new FormData();
                formData.append('file', file);
                formData.append('session_id', sessionId);

                const uploadResponse = await fetch(`${backendUrl}/upload/`, {
                    method: 'POST',
                    body: formData,
                });

                if (!uploadResponse.ok) {
                    throw new Error(`Upload failed for ${file.name}`);
                }

                const responseData = await uploadResponse.json();
                const { task_id } = responseData;

                if (!task_id) {
                    throw new Error(`No task_id returned for ${file.name}`);
                }

                // Update processingFiles with the new task_id
                setProcessingFiles(prev => ({
                    ...prev,
                    [task_id]: 'queued',
                }));

                // Start checking the processing status
                checkProcessingStatus(task_id, file);
            }
            setUploadSuccess(true);
            setUnprocessedFiles([]);
        } catch (error: unknown) {
            console.error('Upload processing error:', error);
            const errorMessage = error instanceof Error 
                ? error.message 
                : 'An unexpected error occurred';
            setError(`Failed to upload files: ${errorMessage}`);
            setUploadSuccess(false);
        } finally {
            setUploadLoading(false);
        }
    }, [unprocessedFiles, sessionId]);

    // Define checkProcessingStatus
    const checkProcessingStatus = useCallback(async (taskId: string, file: File) => {
        try {
            const response = await fetch(`${backendUrl}/process_status/${taskId}`);
            const data = await response.json();

            if (data.status === 'completed') {
                setProcessingFiles(prev => {
                    const newState = { ...prev };
                    delete newState[taskId];
                    return newState;
                });
                setProcessedFiles(prev => [...prev, file]);
                setUploadSuccess(true);
            } else if (data.status === 'failed') {
                setError(`Processing failed for ${file.name}`);
                setProcessingFiles(prev => {
                    const newState = { ...prev };
                    delete newState[taskId];
                    return newState;
                });
            } else {
                // Continue checking if still processing
                setTimeout(() => checkProcessingStatus(taskId, file), 2000);
            }
        } catch (error: unknown) {
            console.error('Status check error:', error);
            setError('Error checking processing status');
        }
    }, []);

    // Monitor Processing Status
    useEffect(() => {
        Object.entries(processingFiles).forEach(([taskId, status]) => {
            if (status === 'queued' || status === 'processing') {
                // Find the file associated with the taskId
                const file = unprocessedFiles.find(f => f.name === taskId); // Adjust this logic based on your mapping
                if (file) {
                    checkProcessingStatus(taskId, file);
                }
            }
        });
    }, [processingFiles, checkProcessingStatus, unprocessedFiles]);

    // Handle Sending Messages
    const handleSendMessage = useCallback(async (e?: React.FormEvent) => {
        if (e) e.preventDefault();

        if (!input.trim()) {
            setError('Please enter a question');
            return;
        }

        if (!uploadSuccess) {
            setError('Please upload and process PDFs first');
            return;
        }

        if (questionLoading) {
            // Prevent sending another message while loading
            return;
        }

        const currentInput = input.trim();
        setInput('');
        setQuestionLoading(true);
        setError('');

        const userMessage: Message = { 
            id: uuidv4(),
            role: 'user', 
            content: currentInput, 
            timestamp: new Date().toLocaleTimeString() 
        };

        setMessages(prev => [...prev, userMessage]);
        setShouldAutoScroll(true);

        try {
            const response = await fetch(`${backendUrl}/ask/`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ 
                    text: currentInput,
                    session_id: sessionId
                }),
            });

            if (!response.ok) throw new Error('Failed to get answer');

            const data = await response.json();

            const assistantMessage: Message = {
                id: uuidv4(),
                role: 'assistant',
                content: data.answer,
                timestamp: new Date().toLocaleTimeString()
            };

            setMessages(prev => [...prev, assistantMessage]);
        } catch (error: unknown) {
            console.error('Error during question processing:', error);
            const errorMessage = error instanceof Error 
                ? error.message 
                : 'An unexpected error occurred';
            setError(`Failed to get answer: ${errorMessage}`);
        } finally {
            setQuestionLoading(false);
        }
    }, [input, uploadSuccess, questionLoading, sessionId]);

    // Handle Clearing Conversation
    const handleClearConversation = useCallback(async () => {
        try {
            const response = await fetch(`${backendUrl}/clear_conversation/`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ session_id: sessionId }),
            });

            if (!response.ok) {
                throw new Error('Failed to clear conversation');
            }

            setMessages([]);
            setShouldAutoScroll(true);
        } catch (error: unknown) {
            console.error('Clear conversation error:', error);
            const errorMessage = error instanceof Error 
                ? error.message 
                : 'An unexpected error occurred';
            setError(`Failed to clear conversation history: ${errorMessage}`);
        }
    }, [sessionId]);

    return (
        <div className="flex flex-col h-screen bg-gray-100">
            <header className="bg-white shadow-sm z-10">
                <div className="max-w-7xl mx-auto py-4 px-4 sm:px-6 lg:px-8">
                    <h1 className="text-lg font-semibold text-gray-900">PDF Chat Application</h1>
                </div>
            </header>
            
            <main className="flex-1 overflow-auto">
                <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-full">
                    <div className="flex flex-col lg:flex-row min-h-screen py-8 gap-4 lg:gap-8">
                        {/* Left column - Chat */}
                        <div className="h-[80vh] lg:flex-1 flex flex-col bg-white rounded-lg shadow min-h-0">
                            <div className="p-6 flex flex-col h-full">
                                <div className="flex justify-between items-center mb-4">
                                    <h2 className="text-lg font-medium text-gray-900">Chat</h2>
                                    <Button 
                                        variant="outline" 
                                        size="sm" 
                                        onClick={handleClearConversation}
                                        aria-label="Clear Conversation"
                                    >
                                        Clear Conversation
                                    </Button>
                                </div>

                                <div 
                                    ref={chatContainerRef}
                                    className="overflow-y-auto pr-4 mb-4 min-h-0"
                                    aria-label="Chat Messages"
                                >
                                    {messages.length === 0 ? (
                                        <p className="text-center text-gray-500 mt-8">Your chat messages will appear here.</p>
                                    ) : (
                                        <div className="space-y-4">
                                            {messages.map((message) => (
                                                <ChatMessage key={message.id} message={message} />
                                            ))}
                                            {questionLoading && (
                                                <div className="flex justify-start mb-4">
                                                    <Avatar className="mr-2">
                                                        <AvatarFallback>AI</AvatarFallback>
                                                    </Avatar>
                                                    <div className="rounded-lg p-4 bg-gray-200">
                                                        <div className="animate-pulse">Thinking...</div>
                                                    </div>
                                                </div>
                                            )}
                                        </div>
                                    )}
                                    {/* Dummy div for scrolling */}
                                    <div ref={messagesEndRef} />
                                </div>

                                <form onSubmit={handleSendMessage} className="mt-auto">
                                    <div className="flex items-center gap-2">
                                        <Input
                                            id="chat-input"
                                            type="text"
                                            placeholder="Ask a question about your documents..."
                                            value={input}
                                            onChange={(e) => setInput(e.target.value)}
                                            disabled={questionLoading || !uploadSuccess}
                                            className="flex-grow"
                                            ref={inputRef}
                                            aria-label="Chat Input"
                                        />
                                        <Button 
                                            type="submit"
                                            disabled={questionLoading || !uploadSuccess || !input.trim()}
                                            className="flex items-center gap-2"
                                            aria-label="Send Message"
                                        >
                                            {questionLoading ? (
                                                <Loader2 className="h-4 w-4 animate-spin" aria-hidden="true" />
                                            ) : (
                                                <Send className="h-4 w-4" aria-hidden="true" />
                                            )}
                                            Send
                                        </Button>
                                    </div>
                                </form>
                            </div>
                        </div>

                        {/* Right column - Document Upload */}
                        <div className="w-full lg:w-1/3 flex flex-col">
                            <div className="flex-1 min-h-0">
                                <section aria-labelledby="documents-section" className="bg-white rounded-lg shadow p-6 h-full">
                                    <h2 id="documents-section" className="text-lg font-medium text-gray-900 mb-4">Documents</h2>
                                    <div className="space-y-4">
                                        <FileDropZone
                                            onFileSelect={handleFileSelect}
                                            unprocessedFiles={unprocessedFiles}
                                            isUploading={uploadLoading}
                                        />
                                        
                                        <Button
                                            onClick={handleUpload}
                                            disabled={unprocessedFiles.length === 0 || uploadLoading || Object.keys(processingFiles).length > 0}
                                            className="w-full"
                                            aria-label="Upload and Process Files"
                                        >
                                            {uploadLoading ? (
                                            <>
                                                <Loader2 className="w-5 h-5 animate-spin mr-2" aria-hidden="true" />
                                                Uploading...
                                            </>
                                            ) : Object.keys(processingFiles).length > 0 ? (
                                            <>
                                                <Loader2 className="w-5 h-5 animate-spin mr-2" aria-hidden="true" />
                                                Processing Files...
                                            </>
                                            ) : (
                                            <>
                                                <Upload className="w-5 h-5 mr-2" aria-hidden="true" />
                                                Upload & Process
                                            </>
                                            )}
                                        </Button>

                                        {/* Processing Status */}
                                        {Object.keys(processingFiles).length > 0 && (
                                            <div className="text-sm text-gray-500">
                                                Processing {Object.keys(processingFiles).length} file(s)...
                                            </div>
                                        )}

                                        {/* Success Message */}
                                        {uploadSuccess && Object.keys(processingFiles).length === 0 && (
                                            <div className="p-4 bg-green-50 text-green-700 rounded-xl flex items-center gap-2">
                                                <CheckCircle className="w-5 h-5" aria-hidden="true" />
                                                <span>All files processed successfully!</span>
                                            </div>
                                        )}

                                        {/* Error Display */}
                                        {error && (
                                            <div className="p-4 bg-red-50 text-red-700 rounded-xl flex items-center gap-2">
                                                <X className="w-5 h-5" aria-hidden="true" />
                                                <span>{error}</span>
                                            </div>
                                        )}

                                        {/* Processed Files List */}
                                        {processedFiles.length > 0 && (
                                            <div 
                                                className="overflow-y-auto mt-4"
                                                style={{ maxHeight: '200px' }}
                                                aria-label="Processed Files List"
                                            >
                                                <div className="space-y-2">
                                                    {processedFiles.map((file, index) => (
                                                        <div key={index} className="flex items-center gap-2 p-2 rounded-lg bg-gray-50">
                                                            <FileIcon className="w-5 h-5 text-blue-600" aria-hidden="true" />
                                                            <div className="flex-1 min-w-0">
                                                                <p className="text-sm font-medium text-gray-900 truncate">
                                                                    {file.name}
                                                                </p>
                                                                <p className="text-sm text-gray-500">
                                                                    {(file.size / 1024).toFixed(2)} KB
                                                                </p>
                                                            </div>
                                                        </div>
                                                    ))}
                                                </div>
                                            </div>
                                        )}
                                    </div>
                                </section>
                            </div>
                        </div>
                    </div>
                </div>
            </main>
        </div>
    );

};

export default RAGInterface;
