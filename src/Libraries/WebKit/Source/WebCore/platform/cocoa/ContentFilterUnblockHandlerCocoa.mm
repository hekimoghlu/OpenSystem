/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 10, 2022.
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
#import "config.h"
#import "ContentFilterUnblockHandler.h"

#if ENABLE(CONTENT_FILTERING)

#import "ContentFilter.h"
#import "Logging.h"
#import "ResourceRequest.h"
#import <pal/spi/cocoa/NSKeyedUnarchiverSPI.h>
#import <pal/spi/cocoa/WebFilterEvaluatorSPI.h>
#import <wtf/BlockObjCExceptions.h>
#import <wtf/SoftLinking.h>
#import <wtf/cocoa/VectorCocoa.h>
#import <wtf/text/CString.h>

#if PLATFORM(IOS_FAMILY)
#import "WebCoreThreadRun.h"
#endif

#if HAVE(PARENTAL_CONTROLS_WITH_UNBLOCK_HANDLER)
SOFT_LINK_PRIVATE_FRAMEWORK(WebContentAnalysis);
SOFT_LINK_CLASS(WebContentAnalysis, WebFilterEvaluator);
#endif

namespace WebCore {

ContentFilterUnblockHandler::ContentFilterUnblockHandler(String unblockURLHost, UnblockRequesterFunction unblockRequester)
    : m_unblockURLHost { WTFMove(unblockURLHost) }
    , m_unblockRequester { WTFMove(unblockRequester) }
{
    LOG(ContentFiltering, "Creating ContentFilterUnblockHandler with an unblock requester and unblock URL host <%s>.\n", m_unblockURLHost.ascii().data());
}

#if HAVE(PARENTAL_CONTROLS_WITH_UNBLOCK_HANDLER)
ContentFilterUnblockHandler::ContentFilterUnblockHandler(String unblockURLHost, RetainPtr<WebFilterEvaluator> evaluator)
    : m_unblockURLHost { WTFMove(unblockURLHost) }
    , m_webFilterEvaluator { WTFMove(evaluator) }
{
    LOG(ContentFiltering, "Creating ContentFilterUnblockHandler with a WebFilterEvaluator and unblock URL host <%s>.\n", m_unblockURLHost.ascii().data());
}
#endif

ContentFilterUnblockHandler::ContentFilterUnblockHandler(
    String&& unblockURLHost,
    URL&& unreachableURL,
#if HAVE(PARENTAL_CONTROLS_WITH_UNBLOCK_HANDLER)
    Vector<uint8_t>&& webFilterEvaluatorData,
#endif
    bool unblockedAfterRequest
    ) : m_unblockURLHost(WTFMove(unblockURLHost))
    , m_unreachableURL(WTFMove(unreachableURL))
#if HAVE(PARENTAL_CONTROLS_WITH_UNBLOCK_HANDLER)
    , m_webFilterEvaluator(unpackWebFilterEvaluatorData(WTFMove(webFilterEvaluatorData)))
#endif
    , m_unblockedAfterRequest(unblockedAfterRequest)
{
}

#if HAVE(PARENTAL_CONTROLS_WITH_UNBLOCK_HANDLER)
// FIXME: Remove the conversion to and from Vector<uint8_t> and serialize individual members when rdar://107281862 is resolved.
Vector<uint8_t> ContentFilterUnblockHandler::webFilterEvaluatorData() const
{
    NSError *error = nil;
    return makeVector([NSKeyedArchiver archivedDataWithRootObject:m_webFilterEvaluator.get() requiringSecureCoding:YES error:&error]);
}

RetainPtr<WebFilterEvaluator> ContentFilterUnblockHandler::unpackWebFilterEvaluatorData(Vector<uint8_t>&& vector)
{
    NSError *error { nil };
    NSSet<Class> *classes = [NSSet setWithObjects:getWebFilterEvaluatorClass(), NSNumber.class, NSURL.class, NSString.class, NSMutableString.class, nil];
    NSData *data = [NSData dataWithBytesNoCopy:vector.data() length:vector.size() freeWhenDone:NO];
    return [NSKeyedUnarchiver _strictlyUnarchivedObjectOfClasses:classes fromData:data error:&error];
}
#endif

void ContentFilterUnblockHandler::wrapWithDecisionHandler(const DecisionHandlerFunction& decisionHandler)
{
    ContentFilterUnblockHandler wrapped { *this };
    UnblockRequesterFunction wrappedRequester { [wrapped, decisionHandler](DecisionHandlerFunction wrappedDecisionHandler) {
        wrapped.requestUnblockAsync([wrappedDecisionHandler, decisionHandler](bool unblocked) {
            wrappedDecisionHandler(unblocked);
            decisionHandler(unblocked);
        });
    }};
#if HAVE(PARENTAL_CONTROLS_WITH_UNBLOCK_HANDLER)
    m_webFilterEvaluator = nullptr;
#endif
    std::swap(m_unblockRequester, wrappedRequester);
}

bool ContentFilterUnblockHandler::needsUIProcess() const
{
#if HAVE(PARENTAL_CONTROLS_WITH_UNBLOCK_HANDLER)
    return m_webFilterEvaluator;
#else
    return false;
#endif
}

bool ContentFilterUnblockHandler::canHandleRequest(const ResourceRequest& request) const
{
    if (!m_unblockRequester && !m_unblockedAfterRequest) {
#if HAVE(PARENTAL_CONTROLS_WITH_UNBLOCK_HANDLER)
        if (!m_webFilterEvaluator)
            return false;
#else
        return false;
#endif
    }

    bool isUnblockRequest = request.url().protocolIs(ContentFilter::urlScheme()) && equalIgnoringASCIICase(request.url().host(), m_unblockURLHost);
#if !LOG_DISABLED
    if (isUnblockRequest)
        LOG(ContentFiltering, "ContentFilterUnblockHandler will handle <%s> as an unblock request.\n", request.url().string().ascii().data());
#endif
    return isUnblockRequest;
}

void ContentFilterUnblockHandler::requestUnblockAsync(DecisionHandlerFunction decisionHandler) const
{
#if HAVE(PARENTAL_CONTROLS_WITH_UNBLOCK_HANDLER)
    if (m_webFilterEvaluator) {
        [m_webFilterEvaluator unblockWithCompletion:[decisionHandler](BOOL unblocked, NSError *) {
            callOnMainThread([decisionHandler, unblocked] {
                LOG(ContentFiltering, "WebFilterEvaluator %s the unblock request.\n", unblocked ? "allowed" : "did not allow");
                decisionHandler(unblocked);
            });
        }];
        return;
    }
#endif
    auto unblockRequester = m_unblockRequester;
    if (!unblockRequester && m_unblockedAfterRequest) {
        unblockRequester = [unblocked = m_unblockedAfterRequest](ContentFilterUnblockHandler::DecisionHandlerFunction function) {
            function(unblocked);
        };
    }
    if (unblockRequester) {
        unblockRequester([decisionHandler](bool unblocked) {
            callOnMainThread([decisionHandler, unblocked] {
                decisionHandler(unblocked);
            });
        });
    } else {
        callOnMainThread([decisionHandler] {
            auto unblocked = false;
            decisionHandler(unblocked);
        });
    }
}

void ContentFilterUnblockHandler::setUnblockedAfterRequest(bool unblocked)
{
    m_unblockedAfterRequest = unblocked;
}

} // namespace WebCore

#endif // ENABLE(CONTENT_FILTERING)
