/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 15, 2023.
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
#include "config.h"
#include "WebMemorySampler.h"

#if ENABLE(MEMORY_SAMPLER)

#include <stdio.h>
#include <wtf/ProcessID.h>
#include <wtf/text/CString.h>
#include <wtf/text/MakeString.h>
#include <wtf/text/StringBuilder.h>

namespace WebKit {
using namespace WebCore;

static const char separator = '\t';

static void appendSpaces(StringBuilder& string, int count)
{
    for (int i = 0; i < count; ++i)
        string.append(' ');
}

WebMemorySampler* WebMemorySampler::singleton()
{
    static WebMemorySampler* sharedMemorySampler;
    if (!sharedMemorySampler)
        sharedMemorySampler = new WebMemorySampler();
    return sharedMemorySampler;
}

WebMemorySampler::WebMemorySampler() 
    : m_sampleTimer(*this, &WebMemorySampler::sampleTimerFired)
    , m_stopTimer(*this, &WebMemorySampler::stopTimerFired)
    , m_isRunning(false)
    , m_runningTime(0)
{
}

void WebMemorySampler::start(const double interval) 
{
    if (m_isRunning) 
        return;
    
    initializeTempLogFile();
    initializeTimers(interval);
}

void WebMemorySampler::start(SandboxExtension::Handle&& sampleLogFileHandle, const String& sampleLogFilePath, const double interval)
{
    if (m_isRunning) 
        return;
    
    // If we are on a system without SandboxExtension the handle and filename will be empty
    if (sampleLogFilePath.isEmpty()) {
        start(interval);
        return;
    }
        
    initializeSandboxedLogFile(WTFMove(sampleLogFileHandle), sampleLogFilePath);
    initializeTimers(interval);
   
}

void WebMemorySampler::initializeTimers(double interval)
{
    m_sampleTimer.startRepeating(1_s);
    SAFE_PRINTF("Started memory sampler for process %s %d", processName().utf8(), getCurrentProcessID());
    if (interval > 0) {
        m_stopTimer.startOneShot(1_s * interval);
        printf(" for a interval of %g seconds", interval);
    }
    SAFE_PRINTF("; Sampler log file stored at: %s\n", m_sampleLogFilePath.utf8());
    m_runningTime = interval;
    m_isRunning = true;
}

void WebMemorySampler::stop() 
{
    if (!m_isRunning) 
        return;
    m_sampleTimer.stop();
    FileSystem::closeFile(m_sampleLogFile);

    SAFE_PRINTF("Stopped memory sampler for process %s %d\n", processName().utf8(), getCurrentProcessID());
    // Flush stdout buffer so python script can be guaranteed to read up to this point.
    fflush(stdout);
    m_isRunning = false;
    
    if (m_stopTimer.isActive())
        m_stopTimer.stop();
    
    if (m_sampleLogSandboxExtension) {
        m_sampleLogSandboxExtension->revoke();
        m_sampleLogSandboxExtension = nullptr;
    }    
}

bool WebMemorySampler::isRunning() const
{
    return m_isRunning;
}
    
void WebMemorySampler::initializeTempLogFile()
{
    auto result = FileSystem::openTemporaryFile(processName());
    m_sampleLogFilePath = result.first;
    m_sampleLogFile = result.second;
    writeHeaders();
}

void WebMemorySampler::initializeSandboxedLogFile(SandboxExtension::Handle&& sampleLogSandboxHandle, const String& sampleLogFilePath)
{
    m_sampleLogSandboxExtension = SandboxExtension::create(WTFMove(sampleLogSandboxHandle));
    if (m_sampleLogSandboxExtension)
        m_sampleLogSandboxExtension->consume();
    m_sampleLogFilePath = sampleLogFilePath;
    m_sampleLogFile = FileSystem::openFile(m_sampleLogFilePath, FileSystem::FileOpenMode::Truncate);
    writeHeaders();
}

void WebMemorySampler::writeHeaders()
{
    auto processDetails = makeString("Process: "_s, processName(), " Pid: "_s, getCurrentProcessID(), '\n').utf8();
    FileSystem::writeToFile(m_sampleLogFile, processDetails.span());
}

void WebMemorySampler::sampleTimerFired()
{
    sendMemoryPressureEvent();
    appendCurrentMemoryUsageToFile(m_sampleLogFile);
}

void WebMemorySampler::stopTimerFired()
{
    if (!m_isRunning)
        return;
    printf("%g seconds elapsed. Stopping memory sampler...\n", m_runningTime);
    stop();
}

void WebMemorySampler::appendCurrentMemoryUsageToFile(FileSystem::PlatformFileHandle&)
{
    // Collect statistics from allocators and get RSIZE metric
    StringBuilder statString;
    WebMemoryStatistics memoryStats = sampleWebKit();

    if (!memoryStats.values.isEmpty()) {
        statString.append(separator);
        for (size_t i = 0; i < memoryStats.values.size(); ++i) {
            statString.append('\n', separator, memoryStats.keys[i]);
            appendSpaces(statString, 35 - memoryStats.keys[i].length());
            statString.append(memoryStats.values[i]);
        }
    }
    statString.append('\n');

    CString utf8String = statString.toString().utf8();
    FileSystem::writeToFile(m_sampleLogFile, utf8String.span());
}

}

#endif
