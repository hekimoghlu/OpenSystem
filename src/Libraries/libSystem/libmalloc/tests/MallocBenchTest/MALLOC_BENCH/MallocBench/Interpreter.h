/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 18, 2024.
 *
 * Licensed under the Apache License, Version 2.0 (the ""License"");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at:
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an ""AS IS"" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Please contact NeXTHub Corporation, 651 N Broad St, Suite 201, 
 * Middletown, DE 19709, New Castle County, USA.
 *
 */
#ifndef Interpreter_h
#define Interpreter_h

#include <condition_variable>
#include <mutex>
#include <thread>
#include <vector>

class Interpreter {
public:
    Interpreter(const char* fileName, bool shouldFreeAllObjects = true, bool useThreadId = false);
    ~Interpreter();

    void run();
    void detailedReport();

private:
    typedef unsigned short ThreadId; // 0 is the main thread
    typedef unsigned short Log2Alignment; // log2(alignment) or ~0 for non power of 2.
    enum Opcode { op_malloc, op_free, op_realloc, op_align_malloc };
    struct Op { Opcode opcode; ThreadId threadId; Log2Alignment alignLog2; size_t slot; size_t size; };
    struct Record { void* object; size_t size; };

    class Thread
    {
    public:
        Thread(Interpreter*, ThreadId);
        ~Thread();

        void runThread();

        void waitToRun();
        void switchTo();
        void stop();
        
        bool isMainThread() { return m_threadId == 0; }

    private:
        ThreadId m_threadId;
        Interpreter* m_myInterpreter;
        std::condition_variable m_shouldRun;
        std::thread m_thread;
    };

    bool readOps();
    void doOnSameThread(ThreadId);
    void switchToThread(ThreadId);

    void doMallocOp(Op, ThreadId);
    
    bool m_shouldFreeAllObjects;
    bool m_useThreadId;
    int m_fd;
    size_t m_opCount;
    size_t m_remaining;
    size_t m_opsCursor;
    size_t m_opsInBuffer;
    ThreadId m_currentThreadId;
    std::vector<Op> m_ops;
    std::mutex m_threadMutex;
    std::condition_variable m_shouldRun;
    std::vector<Thread*> m_threads;
    std::vector<Record> m_objects;
};

#endif // Interpreter_h
