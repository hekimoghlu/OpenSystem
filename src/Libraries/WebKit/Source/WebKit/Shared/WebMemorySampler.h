/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 17, 2025.
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
#ifndef WebMemorySampler_h
#define WebMemorySampler_h

#if ENABLE(MEMORY_SAMPLER)

#include "SandboxExtension.h"
#include <WebCore/Timer.h>
#include <wtf/FileSystem.h>
#include <wtf/Noncopyable.h>
#include <wtf/RefPtr.h>
#include <wtf/Vector.h>
#include <wtf/text/WTFString.h>

namespace WebKit {

struct SystemMallocStats;

struct WebMemoryStatistics {
    Vector<String> keys;
    Vector<size_t> values;
};
    
class WebMemorySampler {
    WTF_MAKE_NONCOPYABLE(WebMemorySampler);
public:
    static WebMemorySampler* singleton();
    void start(const double interval = 0);
    void start(SandboxExtension::Handle&&, const String&, const double interval = 0);
    void stop();
    bool isRunning() const;
    
    // Do nothing since this is a singleton.
    void ref() const { }
    void deref() const { }

private:
    WebMemorySampler();
    ~WebMemorySampler();
    
    void initializeTempLogFile();
    void initializeSandboxedLogFile(SandboxExtension::Handle&&, const String&);
    void writeHeaders();
    void initializeTimers(double);
    void sampleTimerFired();
    void stopTimerFired();
    void appendCurrentMemoryUsageToFile(FileSystem::PlatformFileHandle&);
    void sendMemoryPressureEvent();
    
    SystemMallocStats sampleSystemMalloc() const;
    size_t sampleProcessCommittedBytes() const;
    WebMemoryStatistics sampleWebKit() const;
    String processName() const;
    
    FileSystem::PlatformFileHandle m_sampleLogFile { FileSystem::invalidPlatformFileHandle };
    String m_sampleLogFilePath;
    WebCore::Timer m_sampleTimer;
    WebCore::Timer m_stopTimer;
    bool m_isRunning;
    double m_runningTime;
    RefPtr<SandboxExtension> m_sampleLogSandboxExtension;
};

}

#endif

#endif
