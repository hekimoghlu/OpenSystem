/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 24, 2022.
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

#include <wtf/CryptographicallyRandomNumber.h>
#include <wtf/Logger.h>

namespace WTF {

class LoggerHelper {
public:
    virtual ~LoggerHelper() = default;

    virtual const Logger& logger() const = 0;
    virtual ASCIILiteral logClassName() const = 0;
    virtual WTFLogChannel& logChannel() const = 0;
    virtual uint64_t logIdentifier() const = 0;

#if !RELEASE_LOG_DISABLED

#define LOGIDENTIFIER WTF::Logger::LogSiteIdentifier(logClassName(), __func__, logIdentifier())

#if VERBOSE_RELEASE_LOG
#define ALWAYS_LOG(...)     Ref { logger() }->logAlwaysVerbose(logChannel(), __FILE__, __func__, __LINE__, __VA_ARGS__)
#define ERROR_LOG(...)      Ref { logger() }->errorVerbose(logChannel(), __FILE__, __func__, __LINE__, __VA_ARGS__)
#define WARNING_LOG(...)    Ref { logger() }->warningVerbose(logChannel(), __FILE__, __func__, __LINE__, __VA_ARGS__)
#define INFO_LOG(...)       Ref { logger() }->infoVerbose(logChannel(), __FILE__, __func__, __LINE__, __VA_ARGS__)
#define DEBUG_LOG(...)      Ref { logger() }->debugVerbose(logChannel(), __FILE__, __func__, __LINE__, __VA_ARGS__)
#else
#define ALWAYS_LOG(...)     Ref { logger() }->logAlways(logChannel(), __VA_ARGS__)
#define ERROR_LOG(...)      Ref { logger() }->error(logChannel(), __VA_ARGS__)
#define WARNING_LOG(...)    Ref { logger() }->warning(logChannel(), __VA_ARGS__)
#define INFO_LOG(...)       Ref { logger() }->info(logChannel(), __VA_ARGS__)
#define DEBUG_LOG(...)      Ref { logger() }->debug(logChannel(), __VA_ARGS__)

#define ALWAYS_LOG_WITH_THIS(thisPtr, ...)     Ref { thisPtr->logger() }->logAlways(thisPtr->logChannel(), __VA_ARGS__)
#endif

#define WILL_LOG(_level_)   Ref { logger() }->willLog(logChannel(), _level_)

#define ALWAYS_LOG_IF(condition, ...)     if (condition) ALWAYS_LOG(__VA_ARGS__)
#define ERROR_LOG_IF(condition, ...)      if (condition) ERROR_LOG(__VA_ARGS__)
#define WARNING_LOG_IF(condition, ...)    if (condition) WARNING_LOG(__VA_ARGS__)
#define INFO_LOG_IF(condition, ...)       if (condition) INFO_LOG(__VA_ARGS__)
#define DEBUG_LOG_IF(condition, ...)      if (condition) DEBUG_LOG(__VA_ARGS__)

#define ALWAYS_LOG_IF_POSSIBLE(...)     if (RefPtr logger = loggerPtr()) logger->logAlways(logChannel(), __VA_ARGS__)
#define ERROR_LOG_IF_POSSIBLE(...)      if (RefPtr logger = loggerPtr()) logger->error(logChannel(), __VA_ARGS__)
#define WARNING_LOG_IF_POSSIBLE(...)    if (RefPtr logger = loggerPtr()) logger->warning(logChannel(), __VA_ARGS__)
#define INFO_LOG_IF_POSSIBLE(...)       if (RefPtr logger = loggerPtr()) logger->info(logChannel(), __VA_ARGS__)
#define DEBUG_LOG_IF_POSSIBLE(...)      if (RefPtr logger = loggerPtr()) logger->debug(logChannel(), __VA_ARGS__)
#define WILL_LOG_IF_POSSIBLE(_level_)   if (RefPtr logger = loggerPtr()) logger->willLog(logChannel(), _level_)

#if defined(__OBJC__)
#define OBJC_LOGIDENTIFIER WTF::Logger::LogSiteIdentifier(__PRETTY_FUNCTION__, self.logIdentifier)
#define OBJC_ALWAYS_LOG(...)     if (self.loggerPtr && self.logChannel) self.loggerPtr->logAlways(*self.logChannel, __VA_ARGS__)
#define OBJC_ERROR_LOG(...)      if (self.loggerPtr && self.logChannel) self.loggerPtr->error(*self.logChannel, __VA_ARGS__)
#define OBJC_WARNING_LOG(...)    if (self.loggerPtr && self.logChannel) self.loggerPtr->warning(*self.logChannel, __VA_ARGS__)
#define OBJC_INFO_LOG(...)       if (self.loggerPtr && self.logChannel) self.loggerPtr->info(*self.logChannel, __VA_ARGS__)
#define OBJC_DEBUG_LOG(...)      if (self.loggerPtr && self.logChannel) self.loggerPtr->debug(*self.logChannel, __VA_ARGS__)
#endif

    static uint64_t childLogIdentifier(uint64_t parentIdentifier, uint64_t childIdentifier)
    {
        static constexpr uint64_t parentMask = 0xffffffffffff0000ull;
        static constexpr uint64_t maskLowerWord = 0xffffull;
        return reinterpret_cast<uint64_t>((parentIdentifier & parentMask) | (childIdentifier & maskLowerWord));
    }

    static uint64_t uniqueLogIdentifier()
    {
        return cryptographicallyRandomNumber<uint64_t>();
    }

#else // RELEASE_LOG_DISABLED

#define LOGIDENTIFIER (std::nullopt)

#define ALWAYS_LOG(channelName, ...)  (UNUSED_PARAM(channelName))
#define ERROR_LOG(channelName, ...)   (UNUSED_PARAM(channelName))
#define WARNING_LOG(channelName, ...) (UNUSED_PARAM(channelName))
#define NOTICE_LOG(channelName, ...)  (UNUSED_PARAM(channelName))
#define INFO_LOG(channelName, ...)    (UNUSED_PARAM(channelName))
#define DEBUG_LOG(channelName, ...)   (UNUSED_PARAM(channelName))
#define WILL_LOG(_level_)    false

#define ALWAYS_LOG_IF(condition, ...)     ((void)0)
#define ERROR_LOG_IF(condition, ...)      ((void)0)
#define WARNING_LOG_IF(condition, ...)    ((void)0)
#define INFO_LOG_IF(condition, ...)       ((void)0)
#define DEBUG_LOG_IF(condition, ...)      ((void)0)

#define ALWAYS_LOG_IF_POSSIBLE(...)     ((void)0)
#define ERROR_LOG_IF_POSSIBLE(...)      ((void)0)
#define WARNING_LOG_IF_POSSIBLE(...)    ((void)0)
#define INFO_LOG_IF_POSSIBLE(...)       ((void)0)
#define DEBUG_LOG_IF_POSSIBLE(...)      ((void)0)
#define WILL_LOG_IF_POSSIBLE(_level_)   ((void)0)

#endif // RELEASE_LOG_DISABLED

};

} // namespace WTF

using WTF::LoggerHelper;
