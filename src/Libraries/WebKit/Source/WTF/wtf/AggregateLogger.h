/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 28, 2022.
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

#include <wtf/Algorithms.h>
#include <wtf/HashSet.h>
#include <wtf/Logger.h>
#include <wtf/Ref.h>
#include <wtf/RefPtr.h>

namespace WTF {

class AggregateLogger : public Logger {
    WTF_MAKE_NONCOPYABLE(AggregateLogger);
public:

    static Ref<AggregateLogger> create(const void* owner)
    {
        return adoptRef(*new AggregateLogger(owner));
    }

    void addLogger(const Logger& logger)
    {
        m_loggers.add(&logger);
    }

    void removeLogger(const Logger& logger)
    {
        m_loggers.remove(&logger);
    }

    template<typename... Arguments>
    inline void logAlways(WTFLogChannel& channel, UNUSED_VARIADIC_PARAMS const Arguments&... arguments) const
    {
#if RELEASE_LOG_DISABLED
        // "Standard" WebCore logging goes to stderr, which is captured in layout test output and can generally be a problem
        //  on some systems, so don't allow it.
        UNUSED_PARAM(channel);
#else
        log(channel, WTFLogLevel::Always, arguments...);
#endif
    }

    template<typename... Arguments>
    inline void error(WTFLogChannel& channel, const Arguments&... arguments) const
    {
        log(channel, WTFLogLevel::Error, arguments...);
    }

    template<typename... Arguments>
    inline void warning(WTFLogChannel& channel, const Arguments&... arguments) const
    {
        log(channel, WTFLogLevel::Warning, arguments...);
    }

    template<typename... Arguments>
    inline void info(WTFLogChannel& channel, const Arguments&... arguments) const
    {
        log(channel, WTFLogLevel::Info, arguments...);
    }

    template<typename... Arguments>
    inline void debug(WTFLogChannel& channel, const Arguments&... arguments) const
    {
        log(channel, WTFLogLevel::Debug, arguments...);
    }

    inline bool willLog(const WTFLogChannel& channel, WTFLogLevel level) const
    {
        for (auto& loggers : m_loggers) {
            if (!loggers->willLog(channel, level))
                return false;
        }
        return true;
    }

private:
    AggregateLogger(const void* owner)
        : Logger(owner)
    {
    }

    template<typename... Argument>
    inline void log(WTFLogChannel& channel, WTFLogLevel level, const Argument&... arguments) const
    {
        if (!willLog(channel, level))
            return;

        Logger::log(channel, level, arguments...);

        for (const auto& logger : m_loggers) {
            for (Observer& observer : logger->observers())
                observer.didLogMessage(channel, level, { ConsoleLogValue<Argument>::toValue(arguments)... });
        }
    }

    UncheckedKeyHashSet<RefPtr<const Logger>> m_loggers;
};

} // namespace WTF

using WTF::AggregateLogger;
