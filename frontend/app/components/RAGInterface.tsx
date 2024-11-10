'use client';

import React, { useState } from 'react';
import ReactMarkdown from 'react-markdown';

const backendUrl = process.env.NEXT_PUBLIC_BACKEND_URL;

export default function RAGInterface() {
  const [file, setFile] = useState<File | null>(null);
  const [question, setQuestion] = useState('');
  const [answer, setAnswer] = useState('');
  const [uploadLoading, setUploadLoading] = useState(false);
  const [questionLoading, setQuestionLoading] = useState(false);
  const [error, setError] = useState('');
  const [uploadSuccess, setUploadSuccess] = useState(false);

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = event.target.files?.[0];
    if (selectedFile && selectedFile.type === 'application/pdf') {
      setFile(selectedFile);
      setError('');
    } else {
      setError('Please select a PDF file');
      setFile(null);
    }
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

      if (!response.ok) {
        throw new Error('Upload failed');
      }

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

      if (!response.ok) {
        throw new Error('Failed to get answer');
      }

      const data = await response.json();
      setAnswer(data.answer);
    } catch (err) {
      setError('Failed to get answer: ' + (err instanceof Error ? err.message : 'Unknown error'));
    } finally {
      setQuestionLoading(false);
    }
  };

  return (
    <div className="max-w-2xl mx-auto mt-8 p-6 bg-white rounded-lg shadow">
      <h1 className="text-2xl font-bold text-center mb-6">
        PDF Chat
      </h1>

      <div className="space-y-4">
        {/* File Upload Section */}
        <div className="space-y-2">
          <div className="flex items-center gap-2">
            <input
              type="file"
              accept=".pdf"
              onChange={handleFileChange}
              className="flex-1 p-2 border rounded"
            />
            <button
              onClick={handleUpload}
              disabled={!file || uploadLoading}
              className={`px-4 py-2 rounded ${
                uploadLoading || !file
                  ? 'bg-gray-300 cursor-not-allowed'
                  : 'bg-blue-500 hover:bg-blue-600 text-white'
              }`}
            >
              {uploadLoading ? 'Uploading...' : 'Upload'}
            </button>
          </div>
          
          {uploadSuccess && (
            <div className="p-2 bg-green-100 text-green-700 rounded">
              PDF uploaded successfully!
            </div>
          )}
        </div>

        {/* Question Input Section */}
        <div className="space-y-2">
          <input
            type="text"
            placeholder="Ask a question about the PDF..."
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            disabled={questionLoading || !uploadSuccess}
            className="w-full p-2 border rounded"
          />
          <button
            onClick={handleQuestionSubmit}
            disabled={questionLoading || !uploadSuccess || !question.trim()}
            className={`w-full p-2 rounded ${
              questionLoading || !uploadSuccess || !question.trim()
                ? 'bg-gray-300 cursor-not-allowed'
                : 'bg-blue-500 hover:bg-blue-600 text-white'
            }`}
          >
            {questionLoading ? 'Analyzing the document...' : 'Ask Question'}
          </button>
        </div>

        {/* Error Display */}
        {error && (
          <div className="p-2 bg-red-100 text-red-700 rounded">
            {error}
          </div>
        )}

        {/* Answer Display with Markdown */}
        {answer && (
          <div className="mt-4 p-4 bg-gray-50 rounded">
            <h3 className="font-semibold mb-2">Answer:</h3>
            <ReactMarkdown 
              className="prose prose-blue max-w-none"
              components={{
                h1: ({node, ...props}) => <h1 className="text-2xl font-bold my-4" {...props}/>,
                h2: ({node, ...props}) => <h2 className="text-xl font-bold my-3" {...props}/>,
                h3: ({node, ...props}) => <h3 className="text-lg font-bold my-2" {...props}/>,
                ul: ({node, ...props}) => <ul className="list-disc ml-4 my-2" {...props}/>,
                ol: ({node, ...props}) => <ol className="list-decimal ml-4 my-2" {...props}/>,
                li: ({node, ...props}) => <li className="my-1" {...props}/>,
                p: ({node, ...props}) => <p className="my-2" {...props}/>,
                strong: ({node, ...props}) => <strong className="font-bold text-gray-900" {...props}/>,
                em: ({node, ...props}) => <em className="italic text-gray-800" {...props}/>,
              }}
            >
              {answer}
            </ReactMarkdown>
          </div>
        )}
      </div>
    </div>
  );
}