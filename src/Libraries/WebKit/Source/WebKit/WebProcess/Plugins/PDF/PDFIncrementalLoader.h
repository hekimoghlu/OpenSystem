/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 22, 2022.
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
#pragma once

#if ENABLE(PDF_PLUGIN) && HAVE(INCREMENTAL_PDF_APIS)

#include <wtf/Ref.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/ThreadSafeRefCounted.h>
#include <wtf/ThreadSafeWeakHashSet.h>
#include <wtf/ThreadSafeWeakPtr.h>
#include <wtf/Threading.h>
#include <wtf/threads/BinarySemaphore.h>

OBJC_CLASS PDFDocument;

namespace WebCore {
class NetscapePlugInStreamLoader;
}

namespace WebKit {

class ByteRangeRequest;
class PDFPluginBase;
class PDFPluginStreamLoaderClient;

enum class ByteRangeRequestIdentifierType { };
using ByteRangeRequestIdentifier = ObjectIdentifier<ByteRangeRequestIdentifierType>;
using DataRequestCompletionHandler = Function<void(std::span<const uint8_t>)>;

enum class CheckValidRanges : bool;

class PDFIncrementalLoader : public ThreadSafeRefCountedAndCanMakeThreadSafeWeakPtr<PDFIncrementalLoader> {
    WTF_MAKE_TZONE_ALLOCATED(PDFIncrementalLoader);
    WTF_MAKE_NONCOPYABLE(PDFIncrementalLoader);
    friend class ByteRangeRequest;
    friend class PDFPluginStreamLoaderClient;
public:
    ~PDFIncrementalLoader();

    static Ref<PDFIncrementalLoader> create(PDFPluginBase&);

    void clear();

    void incrementalPDFStreamDidReceiveData(const WebCore::SharedBuffer&);
    void incrementalPDFStreamDidFinishLoading();
    void incrementalPDFStreamDidFail();

    void streamLoaderDidStart(ByteRangeRequestIdentifier, RefPtr<WebCore::NetscapePlugInStreamLoader>&&);

    void receivedNonLinearizedPDFSentinel();

#if !LOG_DISABLED
    void logState(WTF::TextStream&);
#endif

    // Only public for the callbacks
    size_t dataProviderGetBytesAtPosition(std::span<uint8_t> buffer, off_t position);
    void dataProviderGetByteRanges(CFMutableArrayRef buffers, std::span<const CFRange> ranges);

private:
    PDFIncrementalLoader(PDFPluginBase&);

    void threadEntry(Ref<PDFIncrementalLoader>&&);
    void transitionToMainThreadDocument();

    bool documentFinishedLoading() const;

    void appendAccumulatedDataToDataBuffer(ByteRangeRequest&);

    void dataSpanForRange(uint64_t position, size_t count, CheckValidRanges, CompletionHandler<void(std::span<const uint8_t>)>&&) const;
    uint64_t availableDataSize() const;

    void getResourceBytesAtPosition(size_t count, off_t position, DataRequestCompletionHandler&&);
    size_t getResourceBytesAtPositionAfterLoadingComplete(std::span<uint8_t> buffer, off_t position);

    void unconditionalCompleteOutstandingRangeRequests();

    ByteRangeRequest* byteRangeRequestForStreamLoader(WebCore::NetscapePlugInStreamLoader&);
    void forgetStreamLoader(WebCore::NetscapePlugInStreamLoader&);
    void cancelAndForgetStreamLoader(WebCore::NetscapePlugInStreamLoader&);

    std::optional<ByteRangeRequestIdentifier> identifierForLoader(WebCore::NetscapePlugInStreamLoader*);
    void removeOutstandingByteRangeRequest(ByteRangeRequestIdentifier);


    bool requestCompleteIfPossible(ByteRangeRequest&);
    void requestDidCompleteWithBytes(ByteRangeRequest&, size_t byteCount);
    void requestDidCompleteWithAccumulatedData(ByteRangeRequest&, size_t completionSize);

#if !LOG_DISABLED
    size_t incrementThreadsWaitingOnCallback() { return ++m_threadsWaitingOnCallback; }
    size_t decrementThreadsWaitingOnCallback() { return --m_threadsWaitingOnCallback; }

    void incrementalLoaderLog(const String&);
    void logStreamLoader(WTF::TextStream&, WebCore::NetscapePlugInStreamLoader&);
#endif

    class SemaphoreWrapper : public ThreadSafeRefCountedAndCanMakeThreadSafeWeakPtr<SemaphoreWrapper> {
    public:
        static Ref<SemaphoreWrapper> create() { return adoptRef(*new SemaphoreWrapper); }

        void wait() { m_semaphore.wait(); }
        void signal()
        {
            m_wasSignaled = true;
            m_semaphore.signal();
        }
        bool wasSignaled() const { return m_wasSignaled; }

    private:
        SemaphoreWrapper() = default;

        BinarySemaphore m_semaphore;
        std::atomic<bool> m_wasSignaled { false };
    };

    RefPtr<SemaphoreWrapper> createDataSemaphore();

    ThreadSafeWeakPtr<PDFPluginBase> m_plugin;

    RetainPtr<PDFDocument> m_backgroundThreadDocument;
    RefPtr<Thread> m_pdfThread;

    Ref<PDFPluginStreamLoaderClient> m_streamLoaderClient;

    struct RequestData;
    std::unique_ptr<RequestData> m_requestData;

    ThreadSafeWeakHashSet<SemaphoreWrapper> m_dataSemaphores WTF_GUARDED_BY_LOCK(m_wasPDFThreadTerminationRequestedLock);

    Lock m_wasPDFThreadTerminationRequestedLock;
    bool m_wasPDFThreadTerminationRequested WTF_GUARDED_BY_LOCK(m_wasPDFThreadTerminationRequestedLock) { false };

#if !LOG_DISABLED
    std::atomic<size_t> m_threadsWaitingOnCallback { 0 };
    std::atomic<size_t> m_completedRangeRequests { 0 };
    std::atomic<size_t> m_completedNetworkRangeRequests { 0 };
#endif


};

} // namespace WebKit

#endif // ENABLE(PDF_PLUGIN) && HAVE(INCREMENTAL_PDF_APIS)
