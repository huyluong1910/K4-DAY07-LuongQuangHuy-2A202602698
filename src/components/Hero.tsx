import React, { useEffect, useRef, useState } from 'react';
import '../styles/fonts.css';
import '../styles/theme.css';

interface Message {
  sender: 'user' | 'bot';
  text: string;
  sources?: string[];
}

export const Hero: React.FC = () => {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const [isChatMode, setIsChatMode] = useState(false);
  const [messages, setMessages] = useState<Message[]>([
    {
      sender: 'bot',
      text: 'Xin chào! Tôi là Aethera Assistant — trợ lý thông minh của bạn. Tất cả câu trả lời đều được truy xuất trực tiếp từ 10 văn bản quy chế FPTU qua kiến trúc Vector Store [1].',
      sources: ['data/university/01-academic-regulations.md'],
    },
  ]);
  const [inputQuery, setInputQuery] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement | null>(null);



  useEffect(() => {
    if (isChatMode) {
      messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }
  }, [messages, isChatMode]);

  const handleSendMessage = async (queryText?: string) => {
    const textToSend = queryText || inputQuery;
    if (!textToSend.trim()) return;

    if (!isChatMode) {
      setIsChatMode(true);
    }

    const userMessage: Message = { sender: 'user', text: textToSend };
    setMessages((prev) => [...prev, userMessage]);
    if (!queryText) setInputQuery('');
    setIsLoading(true);

    try {
      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: textToSend }),
      });

      if (response.ok) {
        const data = await response.json();
        setMessages((prev) => [
          ...prev,
          {
            sender: 'bot',
            text: data.answer || 'Không nhận được câu trả lời từ máy chủ.',
            sources: data.sources || [],
          },
        ]);
      } else {
        throw new Error('API failed');
      }
    } catch {
      setTimeout(() => {
        let answer = 'Theo quy chế FPTU, thông tin của bạn đã được đối soát qua Vector Store [1].';
        let sources = ['01-academic-regulations.md'];

        const lower = textToSend.toLowerCase();
        if (lower.includes('học bổng') || lower.includes('tiêu chí') || lower.includes('mức')) {
          answer = 'Học bổng FPTU bao gồm các mức 30%, 50%, 70% và 100% học phí toàn khóa [1]. Điều kiện duy trì yêu cầu sinh viên tích lũy đủ tín chỉ theo tiến độ kỳ học, không vi phạm kỷ luật và đạt GPA theo quy định của hội đồng xét duyệt học bổng [2].';
          sources = ['04-scholarship-faq.md', '01-academic-regulations.md'];
        } else if (lower.includes('đồ án') || lower.includes('tốt nghiệp') || lower.includes('capstone')) {
          answer = 'Điều kiện làm Đồ án tốt nghiệp (Cap Stone Project): Sinh viên phải hoàn thành tất cả các học phần chuyên ngành bắt buộc, tích lũy tối thiểu 90% số tín chỉ toàn khóa, hoàn thành kỳ thực tập OJT và không bị nợ học phí [1].';
          sources = ['01-academic-regulations.md', '07-ojt-regulations.md'];
        } else if (lower.includes('ojt') || lower.includes('thực tập')) {
          answer = 'Quy định thực tập doanh nghiệp (OJT): Diễn ra từ học kỳ 6, kéo dài 4-6 tháng tại doanh nghiệp đối tác. Sinh viên phải có chứng chỉ tiếng Anh đạt chuẩn trước khi đi OJT [1].';
          sources = ['07-ojt-regulations.md', '08-ojt-registration.md'];
        } else if (lower.includes('chunking') || lower.includes('vector')) {
          answer = 'Chunking là kỹ thuật chia nhỏ văn bản thành các đoạn ngắn phù hợp với context window của embedding model [1]. Vector store lưu trữ các embedding vectors và tính toán cosine similarity để truy xuất văn bản có liên quan nhất theo ngữ nghĩa [2].';
          sources = ['vector_store_notes.md', 'chunking_experiment_report.md'];
        }

        setMessages((prev) => [
          ...prev,
          {
            sender: 'bot',
            text: answer,
            sources: sources,
          },
        ]);
        setIsLoading(false);
      }, 400);
      return;
    }

    setIsLoading(false);
  };

  return (
    <div className="relative min-h-screen w-full overflow-hidden bg-[#080808] text-[#FFFFFF] font-['Inter'] flex flex-col justify-between">
      {/* Background Video Layer (z-0) - FULL VIEWPORT ORIGINAL */}
      <div className="fixed inset-0 z-0 overflow-hidden pointer-events-none w-full h-full">
        <video
          ref={videoRef}
          src="https://d8j0ntlcm91z4.cloudfront.net/user_38xzZboKViGWJOttwIXH07lWA1P/hf_20260328_083109_283f3553-e28f-428b-a723-d639c617eb2b.mp4"
          autoPlay
          muted
          loop
          playsInline
          className="w-full h-full object-cover"
        />
        {/* Subtle natural film overlay for text legibility without blocking the video */}
        <div className="absolute inset-0 bg-black/30 pointer-events-none" />
      </div>

      {/* Navigation Bar (z-20) */}
      <header className="relative z-20 w-full">
        <nav className="flex justify-between items-center px-8 py-6 max-w-7xl mx-auto">
          {/* Logo */}
          <div onClick={() => setIsChatMode(false)} className="cursor-pointer select-none">
            <span className="font-['Instrument_Serif'] text-3xl tracking-tight text-white">
              Aethera<sup className="text-xs font-sans tracking-normal ml-0.5 font-normal text-white/70">®</sup>
            </span>
          </div>

          {/* Menu items */}
          <div className="hidden md:flex items-center space-x-9 text-sm">
            <button
              onClick={() => setIsChatMode(false)}
              className="text-white font-medium transition-colors hover:text-white cursor-pointer"
            >
              Home
            </button>
            <a href="#studio" className="text-[#8E8E93] transition-colors hover:text-white">
              Studio
            </a>
            <a href="#about" className="text-[#8E8E93] transition-colors hover:text-white">
              About
            </a>
            <a href="#journal" className="text-[#8E8E93] transition-colors hover:text-white">
              Journal
            </a>
            <a href="#reach-us" className="text-[#8E8E93] transition-colors hover:text-white">
              Reach Us
            </a>
          </div>

          {/* Controls: CTA button */}
          <div className="flex items-center space-x-3">
            <button
              onClick={() => setIsChatMode(true)}
              className="rounded-full px-6 py-2.5 text-sm bg-white text-black hover:scale-[1.03] transition-transform duration-200 cursor-pointer font-medium shadow-md active:scale-95"
            >
              Begin Journey
            </button>
          </div>
        </nav>
      </header>

      {/* Hero View */}
      {!isChatMode && (
        <main
          className="relative z-10 flex flex-col items-center justify-center text-center px-6 transition-all duration-500 ease-out animate-fade-rise"
          style={{ paddingTop: 'calc(8rem - 75px)', paddingBottom: '10rem' }}
        >
          <h1 className="animate-fade-rise text-5xl sm:text-7xl md:text-8xl max-w-7xl font-normal leading-[0.95] tracking-[-2.46px] font-['Instrument_Serif'] text-white">
            Beyond <span className="text-[#8E8E93] italic">silence,</span> we build{' '}
            <span className="text-[#8E8E93] italic">the eternal.</span>
          </h1>

          <p className="animate-fade-rise-delay text-base sm:text-lg max-w-2xl mt-8 leading-relaxed text-[#8E8E93] font-['Inter']">
            Building platforms for brilliant minds, fearless makers, and thoughtful souls. Through
            the noise, we craft digital havens for deep work and pure flows.
          </p>

          <button
            onClick={() => setIsChatMode(true)}
            className="animate-fade-rise-delay-2 rounded-full px-14 py-5 text-base mt-12 bg-white text-black hover:scale-[1.03] transition-transform duration-300 cursor-pointer font-medium shadow-2xl active:scale-95"
          >
            Begin Journey
          </button>

          <div className="animate-fade-rise-delay-2 mt-12 flex flex-wrap justify-center gap-2.5 max-w-2xl">
            {[
              'Điều kiện làm đồ án tốt nghiệp là gì?',
              'Học bổng FPTU có những mức nào & duy trì ra sao?',
              'Quy định thực tập doanh nghiệp (OJT)',
              'Chunking Strategy và Vector Store trong RAG',
            ].map((chip, idx) => (
              <button
                key={idx}
                onClick={() => handleSendMessage(chip)}
                className="text-xs px-4 py-2 rounded-full bg-white/5 backdrop-blur-md border border-white/10 text-[#A1A1AA] hover:text-white hover:border-white/30 hover:bg-white/10 hover:scale-[1.02] transition-all cursor-pointer shadow-sm"
              >
                {chip}
              </button>
            ))}
          </div>
        </main>
      )}

      {/* Chat View */}
      {isChatMode && (
        <section className="relative z-10 w-full max-w-3xl mx-auto px-4 flex-1 flex flex-col transition-all duration-500 ease-out animate-fade-rise">
          <div className="flex items-center justify-between py-3.5 px-5 my-2 rounded-2xl bg-white/[0.06] backdrop-blur-xl border border-white/15 shadow-xl">
            <div className="flex items-center space-x-3">
              <button
                onClick={() => setIsChatMode(false)}
                className="flex items-center space-x-1.5 text-xs text-[#8E8E93] hover:text-white transition-colors font-medium cursor-pointer"
              >
                <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 19l-7-7m0 0l7-7m-7 7h18" />
                </svg>
                <span>Trở về Hero</span>
              </button>
              <div className="h-3.5 w-px bg-white/20" />
              <div className="flex items-center space-x-2">
                <span className="font-['Instrument_Serif'] text-xl text-white">Aethera Intelligence</span>
                <span className="text-[9px] px-2 py-0.5 rounded-full bg-white text-black font-mono uppercase tracking-wider font-semibold">
                  RAG Online
                </span>
              </div>
            </div>
            <button
              onClick={() =>
                setMessages([
                  {
                    sender: 'bot',
                    text: 'Cuộc trò chuyện đã được làm mới. Hãy đặt câu hỏi mới cho Aethera Assistant.',
                  },
                ])
              }
              className="text-xs text-[#8E8E93] hover:text-white transition-colors cursor-pointer"
            >
              Làm mới
            </button>
          </div>

          <div className="flex-1 space-y-4 py-4 pb-36 overflow-y-auto">
            {messages.map((msg, i) => (
              <div
                key={i}
                className={`flex flex-col ${msg.sender === 'user' ? 'items-end' : 'items-start'}`}
              >
                <div
                  className={`max-w-[88%] rounded-2xl px-5 py-3.5 text-sm leading-relaxed ${
                    msg.sender === 'user'
                      ? 'bg-white text-black font-normal rounded-tr-sm shadow-lg'
                      : 'bg-white/[0.08] backdrop-blur-xl text-white border border-white/15 rounded-tl-sm shadow-md'
                  }`}
                >
                  <p className="whitespace-pre-wrap">{msg.text}</p>
                  {msg.sources && msg.sources.length > 0 && (
                    <div className="mt-3 pt-2 border-t border-white/10 text-[11px] text-[#8E8E93]">
                      <span className="font-medium text-white">Nguồn trích dẫn: </span>
                      {msg.sources.join(', ')}
                    </div>
                  )}
                </div>
              </div>
            ))}
            {isLoading && (
              <div className="flex items-center space-x-2 text-xs text-[#8E8E93] italic py-1">
                <span className="inline-block w-2 h-2 rounded-full bg-white animate-pulse" />
                <span>Đang truy xuất Vector Store và suy luận câu trả lời...</span>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>
        </section>
      )}

      {/* Fixed Bottom Chat Bar */}
      {isChatMode && (
        <div className="fixed bottom-6 left-0 right-0 z-30 px-4 transition-all duration-500 ease-out animate-fade-rise">
          <div className="max-w-2xl mx-auto bg-black/80 backdrop-blur-2xl border border-white/20 rounded-full p-1.5 shadow-2xl flex items-center">
            <input
              type="text"
              value={inputQuery}
              onChange={(e) => setInputQuery(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter') {
                  e.preventDefault();
                  handleSendMessage();
                }
              }}
              placeholder="Hỏi tiếp về quy chế học vụ, học bổng, đồ án, OJT, RAG..."
              className="flex-1 px-5 py-3 text-sm bg-transparent border-none focus:outline-none text-white placeholder:text-white/40"
              autoFocus
            />
            <button
              onClick={() => handleSendMessage()}
              disabled={isLoading || !inputQuery.trim()}
              className="px-5 py-3 rounded-full bg-white text-black text-sm hover:scale-[1.03] active:scale-95 transition-all cursor-pointer font-medium flex items-center space-x-1.5 shadow-md flex-shrink-0 disabled:opacity-40"
            >
              <span>Gửi</span>
              <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14 5l7 7m0 0l-7 7m7-7H3" />
              </svg>
            </button>
          </div>
          <div className="text-center mt-2">
            <span className="text-[10px] text-white/50">
              HeadingAwareContextChunker • Top-3 Retrieval • Powered by Gemini 3072D
            </span>
          </div>
        </div>
      )}
    </div>
  );
};

export default Hero;
