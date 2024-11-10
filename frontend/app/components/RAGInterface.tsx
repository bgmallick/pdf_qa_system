'use client';

import React, { useState, useEffect, useCallback } from 'react';
import { MessageSquareText, Upload, File, CheckCircle, X, Loader2 } from 'lucide-react';
import dynamic from 'next/dynamic';

// Dynamically import ReactMarkdown with no SSR
const ReactMarkdown = dynamic(() => import('react-markdown'), { ssr: false });

const backendUrl = process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:8000';

// Type definitions
interface MarkdownDisplayProps {
  content: string;
}

interface FileDropZoneProps {
  onFileSelect: (file: File) => void;
  file: File | null;
  isUploading: boolean;
}

// Markdown Display Component
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

// File Drop Zone Component
const FileDropZone: React.FC<FileDropZoneProps> = ({ onFileSelect, file, isUploading }) => {
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
    
    const files = e.dataTransfer.files;
    if (files?.[0]?.type === 'application/pdf') {
      onFileSelect(files[0]);
    }
  }, [onFileSelect]);

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
        onChange={(e: React.ChangeEvent<HTMLInputElement>) => 
          e.target.files?.[0] && onFileSelect(e.target.files[0])
        }
        className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
      />
      
      <div className="space-y-4">
        {file ? (
          <div className="flex items-center justify-center gap-3">
            <File className="w-8 h-8 text-blue-600" />
            <span className="text-gray-700 font-medium">{file.name}</span>
          </div>
        ) : (
          <>
            <Upload className="w-12 h-12 text-blue-600 mx-auto" />
            <div className="space-y-2">
              <p className="text-gray-700 font-medium">
                Drag & drop your PDF here or click to browse
              </p>
              <p className="text-sm text-gray-500">
                (Maximum file size: 10MB)
              </p>
            </div>
          </>
        )}
      </div>

      {isUploading && (
        <div className="absolute inset-0 bg-white bg-opacity-90 flex items-center justify-center">
          <Loader2 className="w-8 h-8 text-blue-600 animate-spin" />
        </div>
      )}
    </div>
  );
};

// Main Component
const RAGInterface: React.FC = () => {
  const [file, setFile] = useState<File | null>(null);
  const [question, setQuestion] = useState('');
  const [answer, setAnswer] = useState('');
  const [uploadLoading, setUploadLoading] = useState(false);
  const [questionLoading, setQuestionLoading] = useState(false);
  const [error, setError] = useState('');
  const [uploadSuccess, setUploadSuccess] = useState(false);

  const handleFileSelect = (selectedFile: File) => {
    setFile(selectedFile);
    setError('');
    setUploadSuccess(false);
  };

  const handleUpload = async () => {
    if (!file) {
      setError('Please select a file first');
      return;
    }

    setUploadLoading(true);
    setError('');
    
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch(`${backendUrl}/upload/`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) throw new Error('Upload failed');

      setUploadSuccess(true);
      setError('');
    } catch (err) {
      setError('Failed to upload file: ' + (err instanceof Error ? err.message : 'Unknown error'));
      setUploadSuccess(false);
    } finally {
      setUploadLoading(false);
    }
  };

  const handleQuestionSubmit = async () => {
    if (!question.trim()) {
      setError('Please enter a question');
      return;
    }

    if (!uploadSuccess) {
      setError('Please upload a PDF first');
      return;
    }

    setQuestionLoading(true);
    setError('');

    try {
      const response = await fetch(`${backendUrl}/ask/`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text: question }),
      });

      if (!response.ok) throw new Error('Failed to get answer');

      const data = await response.json();
      setAnswer(data.answer);
    } catch (err) {
      setError('Failed to get answer: ' + (err instanceof Error ? err.message : 'Unknown error'));
    } finally {
      setQuestionLoading(false);
    }
  };

  return (
    <div className="max-w-4xl mx-auto px-4 py-8 min-h-screen">
      <div className="bg-white bg-opacity-90 backdrop-blur-sm rounded-2xl p-8 shadow-lg">
        {/* Header */}
        <div className="flex items-center justify-center gap-3 mb-8">
          <MessageSquareText className="w-8 h-8 text-blue-600" />
          <h1 className="text-3xl font-bold text-center text-gray-900">
            PDF Chat
          </h1>
        </div>

        <div className="space-y-8">
          {/* File Upload Section */}
          <div className="space-y-4">
            <FileDropZone
              onFileSelect={handleFileSelect}
              file={file}
              isUploading={uploadLoading}
            />
            
            <div className="flex justify-end">
              <button
                onClick={handleUpload}
                disabled={!file || uploadLoading}
                className={`px-6 py-3 rounded-xl transition-colors flex items-center gap-2
                  ${uploadLoading || !file
                    ? 'bg-gray-200 cursor-not-allowed'
                    : 'bg-blue-600 hover:bg-blue-700 text-white'
                  }`}
              >
                {uploadLoading ? (
                  <>
                    <Loader2 className="w-5 h-5 animate-spin" />
                    <span>Uploading...</span>
                  </>
                ) : (
                  <>
                    <Upload className="w-5 h-5" />
                    <span>Upload</span>
                  </>
                )}
              </button>
            </div>
            
            {uploadSuccess && (
              <div className="p-4 bg-green-50 text-green-700 rounded-xl flex items-center gap-2">
                <CheckCircle className="w-5 h-5" />
                <span>PDF uploaded successfully!</span>
              </div>
            )}
          </div>

          {/* Question Input Section */}
          <div className="space-y-4">
            <div className="relative">
              <input
                type="text"
                placeholder="Ask a question about the PDF..."
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                disabled={questionLoading || !uploadSuccess}
                className="w-full p-4 pr-12 border rounded-xl bg-white focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-shadow"
              />
              {question && (
                <button
                  onClick={() => setQuestion('')}
                  className="absolute right-4 top-1/2 -translate-y-1/2 text-gray-400 hover:text-gray-600"
                >
                  <X className="w-5 h-5" />
                </button>
              )}
            </div>
            
            <button
              onClick={handleQuestionSubmit}
              disabled={questionLoading || !uploadSuccess || !question.trim()}
              className={`w-full p-4 rounded-xl transition-colors flex items-center justify-center gap-2
                ${questionLoading || !uploadSuccess || !question.trim()
                  ? 'bg-gray-200 cursor-not-allowed'
                  : 'bg-blue-600 hover:bg-blue-700 text-white'
                }`}
            >
              {questionLoading ? (
                <>
                  <Loader2 className="w-5 h-5 animate-spin" />
                  <span>Analyzing the document...</span>
                </>
              ) : (
                'Ask Question'
              )}
            </button>
          </div>

          {/* Error Display */}
          {error && (
            <div className="p-4 bg-red-50 text-red-700 rounded-xl flex items-center gap-2">
              <X className="w-5 h-5" />
              <span>{error}</span>
            </div>
          )}

          {/* Answer Display */}
          {answer && (
            <div className="mt-8 p-6 bg-gray-50 rounded-xl">
              <h3 className="font-semibold text-lg mb-4">Answer:</h3>
              <MarkdownDisplay content={answer} />
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default RAGInterface;