/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 21, 2022.
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
#ifndef HRTFDatabaseLoader_h
#define HRTFDatabaseLoader_h

#include "HRTFDatabase.h"
#include <memory>
#include <wtf/Lock.h>
#include <wtf/RefCounted.h>
#include <wtf/RefPtr.h>
#include <wtf/Threading.h>

namespace WebCore {

// HRTFDatabaseLoader will asynchronously load the default HRTFDatabase in a new thread.

class HRTFDatabaseLoader : public RefCounted<HRTFDatabaseLoader> {
public:
    // Lazily creates a HRTFDatabaseLoader (if not already created) for the given sample-rate
    // and starts loading asynchronously (when created the first time).
    // Returns the HRTFDatabaseLoader.
    // Must be called from the main thread.
    static Ref<HRTFDatabaseLoader> createAndLoadAsynchronouslyIfNecessary(float sampleRate);

    // Both constructor and destructor must be called from the main thread.
    ~HRTFDatabaseLoader();
    
    // Returns true once the default database has been completely loaded.
    bool isLoaded() const;

    // waitForLoaderThreadCompletion() may be called more than once and is thread-safe.
    void waitForLoaderThreadCompletion();
    
    HRTFDatabase* database() { return m_hrtfDatabase.get(); }

    float databaseSampleRate() const { return m_databaseSampleRate; }
    
private:
    // Both constructor and destructor must be called from the main thread.
    explicit HRTFDatabaseLoader(float sampleRate);
    
    // If it hasn't already been loaded, creates a new thread and initiates asynchronous loading of the default database.
    // This must be called from the main thread.
    void loadAsynchronously();

    // Called in asynchronous loading thread.
    void load();

    std::unique_ptr<HRTFDatabase> m_hrtfDatabase;

    // Holding a m_threadLock is required when accessing m_databaseLoaderThread.
    Lock m_threadLock;
    RefPtr<Thread> m_databaseLoaderThread WTF_GUARDED_BY_LOCK(m_threadLock);

    float m_databaseSampleRate;
};

} // namespace WebCore

#endif // HRTFDatabaseLoader_h
