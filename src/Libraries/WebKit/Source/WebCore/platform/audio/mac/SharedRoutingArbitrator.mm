/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 15, 2023.
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
#import "SharedRoutingArbitrator.h"

#if ENABLE(ROUTING_ARBITRATION) && HAVE(AVAUDIO_ROUTING_ARBITER)

#import "Logging.h"
#import <wtf/LoggerHelper.h>
#import <wtf/NeverDestroyed.h>
#include <wtf/TZoneMallocInlines.h>

#import <pal/cocoa/AVFoundationSoftLink.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(SharedRoutingArbitratorToken);

#define TOKEN_LOGIDENTIFIER(token) WTF::Logger::LogSiteIdentifier(logClassName(), __func__, token.logIdentifier())

UniqueRef<SharedRoutingArbitratorToken> SharedRoutingArbitratorToken::create()
{
    return makeUniqueRef<SharedRoutingArbitratorToken>();
}

uint64_t SharedRoutingArbitratorToken::logIdentifier() const
{
    if (!m_logIdentifier)
        m_logIdentifier = LoggerHelper::uniqueLogIdentifier();

    return m_logIdentifier;
}

SharedRoutingArbitrator& SharedRoutingArbitrator::sharedInstance()
{
    static NeverDestroyed<SharedRoutingArbitrator> instance;
    return instance;
}

bool SharedRoutingArbitrator::isInRoutingArbitrationForToken(const SharedRoutingArbitratorToken& token)
{
    return m_tokens.contains(token);
}

void SharedRoutingArbitrator::beginRoutingArbitrationForToken(const SharedRoutingArbitratorToken& token, AudioSession::CategoryType requestedCategory, ArbitrationCallback&& callback)
{
    ASSERT(!isInRoutingArbitrationForToken(token));

    auto identifier = TOKEN_LOGIDENTIFIER(token);
    ALWAYS_LOG_IF(m_logger, identifier, requestedCategory);

    if (m_setupArbitrationOngoing) {
        ALWAYS_LOG_IF(m_logger, identifier, "enqueing callback, arbitration ongoing");
        m_enqueuedCallbacks.append([this, weakToken = WeakPtr { token }, callback = WTFMove(callback), identifier = WTFMove(identifier)] (RoutingArbitrationError error, DefaultRouteChanged routeChanged) mutable {
            if (error == RoutingArbitrationError::None && weakToken)
                m_tokens.add(*weakToken);

            ALWAYS_LOG_IF(m_logger, identifier, "pending arbitration finished, error = ", error, ", routeChanged = ", routeChanged);
            callback(error, routeChanged);
        });

        return;
    }

    if (m_currentCategory) {
        if (*m_currentCategory >= requestedCategory) {
            m_tokens.add(token);
            ALWAYS_LOG_IF(m_logger, identifier, "ignoring, nothing to change");
            callback(RoutingArbitrationError::None, DefaultRouteChanged::No);
            return;
        }

        ALWAYS_LOG_IF(m_logger, identifier, "leaving current arbitration");
        [[PAL::getAVAudioRoutingArbiterClass() sharedRoutingArbiter] leaveArbitration];
    }

    m_currentCategory = requestedCategory;

    AVAudioRoutingArbitrationCategory arbitrationCategory = AVAudioRoutingArbitrationCategoryPlayback;
    switch (requestedCategory) {
    case AudioSession::CategoryType::MediaPlayback:
        arbitrationCategory = AVAudioRoutingArbitrationCategoryPlayback;
        break;
    case AudioSession::CategoryType::RecordAudio:
        arbitrationCategory = AVAudioRoutingArbitrationCategoryPlayAndRecord;
        break;
    case AudioSession::CategoryType::PlayAndRecord:
        arbitrationCategory = AVAudioRoutingArbitrationCategoryPlayAndRecordVoice;
        break;
    default:
        ASSERT_NOT_REACHED();
    }

    m_setupArbitrationOngoing = true;
    m_enqueuedCallbacks.append([this, weakToken = WeakPtr { token }, callback = WTFMove(callback)] (RoutingArbitrationError error, DefaultRouteChanged routeChanged) mutable {
        if (error == RoutingArbitrationError::None && weakToken)
            m_tokens.add(*weakToken);

        callback(error, routeChanged);
    });

    [[PAL::getAVAudioRoutingArbiterClass() sharedRoutingArbiter] beginArbitrationWithCategory:arbitrationCategory completionHandler:[this, identifier = WTFMove(identifier)](BOOL defaultDeviceChanged, NSError * _Nullable error) mutable {
        callOnMainRunLoop([this, defaultDeviceChanged, error = retainPtr(error), identifier = WTFMove(identifier)] {
            if (error)
                ERROR_LOG(identifier, error.get(), ", routeChanged = ", !!defaultDeviceChanged);

            // FIXME: Do we need to reset sample rate and buffer size if the default device changes?

            ALWAYS_LOG_IF(m_logger, identifier, "arbitration completed, category = ", m_currentCategory ? *m_currentCategory : AudioSession::CategoryType::None, ", default device changed = ", !!defaultDeviceChanged);

            Vector<ArbitrationCallback> callbacks = WTFMove(m_enqueuedCallbacks);
            for (auto& callback : callbacks)
                callback(error ? RoutingArbitrationError::Failed : RoutingArbitrationError::None, defaultDeviceChanged ? DefaultRouteChanged::Yes : DefaultRouteChanged::No);

            m_setupArbitrationOngoing = false;
        });
    }];
}

void SharedRoutingArbitrator::endRoutingArbitrationForToken(const SharedRoutingArbitratorToken& token)
{
    ALWAYS_LOG_IF(m_logger, TOKEN_LOGIDENTIFIER(token));

    m_tokens.remove(token);

    if (!m_tokens.isEmptyIgnoringNullReferences())
        return;

    for (auto& callback : m_enqueuedCallbacks)
        callback(RoutingArbitrationError::Cancelled, DefaultRouteChanged::No);

    m_enqueuedCallbacks.clear();
    m_currentCategory.reset();
    [[PAL::getAVAudioRoutingArbiterClass() sharedRoutingArbiter] leaveArbitration];
}

void SharedRoutingArbitrator::setLogger(const Logger& logger)
{
    if (!m_logger)
        m_logger = &logger;
}

const Logger& SharedRoutingArbitrator::logger()
{
    ASSERT(m_logger);
    return *m_logger.get();
}

WTFLogChannel& SharedRoutingArbitrator::logChannel() const
{
    return LogMedia;
}

}
#endif
