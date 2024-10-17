import React, {
  useState,
  FormEvent,
  ChangeEvent,
  useRef,
  useEffect,
} from "react";
import { FaArrowUp } from "react-icons/fa6";
import { MdOutlineAttachFile } from "react-icons/md";
import { FaStop } from "react-icons/fa6";
import { Message } from "../types";
import Results from "./results";
import { FaDownload, FaTimes } from "react-icons/fa";
import { AiOutlineLoading3Quarters } from "react-icons/ai"; // Spinner Icon

const ChatBox: React.FC = () => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputValue, setInputValue] = useState<string>("");
  const [isLoading, setIsLoading] = useState(false); // For showing the loading spinner
  const [abortController, setAbortController] =
    useState<AbortController | null>(null);
  const [attachedFile, setAttachedFile] = useState<File | null>(null);
  const [submittedFiles, setSubmittedFiles] = useState<File[]>([]);
  const [introMessage, setIntroMessage] = useState<string | null>(null); // New state for introductory message

  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();

    if (!inputValue.trim()) return;

    const userMessage: Message = { type: "user", text: inputValue };
    setMessages((prevMessages) => [...prevMessages, userMessage]);

    if (attachedFile) {
      setSubmittedFiles((prevFiles) => [...prevFiles, attachedFile]);
    }

    setInputValue("");
    setAttachedFile(null);
    setIsLoading(true); // Start loading

    const controller = new AbortController();
    setAbortController(controller);

    try {
      const response = await fetch("http://localhost:5000/product_prices", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          prompt: inputValue,
          limit: 4,
        }),
        signal: controller.signal,
      } as RequestInit); // Explicitly type as RequestInit

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      console.log("Response from server:", data); // <-- Log the response for debugging

      // Set the introductory message
      if (data.message) {
        console.log("Introductory message found:", data.message); // Debug
        setIntroMessage(data.message); // Save the intro message to state
      } else {
        console.log("No introductory message in the response."); // Debug
      }

      const botMessage: Message = data.suppliers
        ? { type: "bot", data: data.suppliers } // Product-related structured response
        : { type: "bot", text: data.response }; // General conversation response

      setMessages((prevMessages) => [...prevMessages, botMessage]);
    } catch (error: unknown) {
      if (error instanceof Error && error.name === "AbortError") {
        console.log("Fetch aborted");
      } else {
        const errorMessage: Message = {
          type: "bot",
          text: "Error retrieving data",
        };
        setMessages((prevMessages) => [...prevMessages, errorMessage]);
      }
    }

    setIsLoading(false); // Stop loading
    setAbortController(null);
  };

  const handleStop = () => {
    if (abortController) {
      abortController.abort();
      setIsLoading(false); // Stop loading when request is aborted
    }
  };

  const textAreaRef = useRef<HTMLTextAreaElement | null>(null);

  const handleInputChange = (e: ChangeEvent<HTMLTextAreaElement>) => {
    setInputValue(e.target.value);
  };

  const handleFileChange = (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setAttachedFile(file);
    }
  };

  const handleAttachClick = () => {
    fileInputRef.current?.click();
  };

  const handleRemoveFile = () => {
    setAttachedFile(null);
  };

  useEffect(() => {
    if (textAreaRef.current) {
      textAreaRef.current.style.height = "auto";
      textAreaRef.current.style.height = `${textAreaRef.current.scrollHeight}px`;
    }
  }, [inputValue]);

  const chatEndRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  return (
    <div className="flex flex-col items-center min-h-screen relative">
      <div className="chat-messages w-[680px] max-h-[80vh] overflow-y-auto mb-20 no-scrollbar">
        {messages.map((msg, index) => (
          <div
            key={index}
            className="flex flex-col justify-start w-full mt-[70px]"
          >
            {msg.type === "bot" && msg.data ? (
              // For product-related JSON response
              <div className="flex flex-col gap-[20px] justify-center items-baseline w-full">
                <div className="flex items-center justify-start w-full gap-5">
                  <div className="w-[32px] h-[32px] rounded-full bg-red-700"></div>
                  {/* Display the introductory message */}
                  {introMessage && (
                    // <div className="p-2 mb-4 text-start bg-gray-200 rounded-lg flex items-center">
                    <p className="bg-transparent">{introMessage}</p>
                    // </div>
                  )}
                </div>
                <div className="flex flex-col items-center w-full">
                  <div className="flex flex-col gap-5 w-full flex-wrap items-start">
                    {msg.data.map((result, i) => (
                      <Results key={i} result={result} />
                    ))}
                  </div>
                </div>
              </div>
            ) : msg.type === "bot" && msg.text ? (
              // For general conversation response
              <div className="flex gap-[20px] justify-start  w-full">
                <div className="w-[32px] h-[32px] rounded-full bg-red-700 flex-shrink-0"></div>
                <div className="p-[12px] text-start bg-gray-300 flex rounded-lg">
                  {msg.text}
                </div>
              </div>
            ) : (
              // User message
              <div className="flex gap-[20px] justify-start items-center w-full">
                <img
                  className="w-[32px] h-[32px]   rounded-full"
                  src="../public/images/Profile_image.jpeg"
                  alt=""
                />
                <div className="p-[12px] text-start bg-gray-300 flex rounded-lg">
                  {msg.text}
                </div>
              </div>
            )}
          </div>
        ))}

        {/* Loading spinner */}
        {isLoading && (
          <div className="flex gap-[20px] justify-start w-full mt-[70px]">
            <div className="w-[32px] h-[32px] rounded-full bg-red-700"></div>
            <div className="flex flex-col gap-5 w-full">
              <div className="flex justify-start items-center text-gray-600 gap-2">
                <AiOutlineLoading3Quarters className="animate-spin" size={24} />
                <p>Please wait as I process your request...</p>{" "}
                {/* Loading text */}
              </div>
            </div>
          </div>
        )}

        <div ref={chatEndRef}></div>
      </div>

      <form
        onSubmit={handleSubmit}
        className="w-[680px] rounded fixed bottom-5 box-border"
      >
        {attachedFile && (
          <div className="flex items-center justify-between w-[15%] ml-6 mb-2 p-2 border border-gray-300 bg-transparent  border-1 absolute bottom-16 rounded-md">
            <div className="truncate ">{attachedFile.name}</div>
            <FaTimes
              className="text-red-500 cursor-pointer"
              onClick={handleRemoveFile}
            />
          </div>
        )}

        <textarea
          value={inputValue}
          onChange={handleInputChange}
          placeholder="Ask about a product or anything else..."
          className="placeholder: outline-none rounded-[40px] overflow-hidden w-full h-full resize-none bg-gray-300 px-14 pr-14 box-border py-4 relative"
          ref={textAreaRef}
          rows={1}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              handleSubmit(e as unknown as FormEvent);
            }
          }}
        />

        {submittedFiles.length > 0 && (
          <div className="mb-2">
            {submittedFiles.map((file, index) => (
              <div
                key={index}
                className="flex items-center justify-between mb-1 p-2 border border-gray-300 bg-gray-100 rounded-md"
              >
                <div className="truncate">{file.name}</div>
                <a href={URL.createObjectURL(file)} download={file.name}>
                  <FaDownload className="text-gray-500 cursor-pointer" />
                </a>
              </div>
            ))}
          </div>
        )}

        <button
          className={`absolute rounded-full h-[32px] w-[32px] bottom-5 right-3 flex justify-center items-center ${
            isLoading
              ? "bg-[#072630]"
              : inputValue || attachedFile
              ? "bg-[#072630]"
              : "bg-gray-400"
          }`}
          type={isLoading ? "button" : "submit"}
          onClick={isLoading ? handleStop : undefined}
          disabled={!inputValue && !attachedFile && !isLoading}
        >
          {isLoading ? (
            <FaStop className="text-white cursor-pointer" />
          ) : (
            <FaArrowUp
              className={`text-white cursor-${
                inputValue || attachedFile ? "pointer" : "not-allowed"
              }`}
            />
          )}
        </button>

        <input
          type="file"
          ref={fileInputRef}
          style={{ display: "none" }}
          onChange={handleFileChange}
        />

        <button
          type="button"
          className="absolute rounded-full h-[32px] w-[32px] bottom-4 left-2 flex justify-center items-center text-lg"
          onClick={handleAttachClick}
        >
          <MdOutlineAttachFile className="text-gray-500 h-[25px] w-[25px] rotate-45 cursor-pointer" />
        </button>
      </form>
    </div>
  );
};

export default ChatBox;
