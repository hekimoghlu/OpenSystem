/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 4, 2023.
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

#include <wtf/TZoneMalloc.h>
#include <wtf/ThreadSafeRefCounted.h>
#include <wtf/Threading.h>
#include <wtf/text/StringHash.h>

namespace PAL {

struct ICUConverterWrapper;

class ThreadGlobalData : public WTF::Thread::ClientData {
    WTF_MAKE_TZONE_ALLOCATED(ThreadGlobalData);
    WTF_MAKE_NONCOPYABLE(ThreadGlobalData);
public:
    PAL_EXPORT virtual ~ThreadGlobalData();

    ICUConverterWrapper& cachedConverterICU() { return *m_cachedConverterICU; }

protected:
    PAL_EXPORT ThreadGlobalData();

private:
    PAL_EXPORT friend ThreadGlobalData& threadGlobalData();

    std::unique_ptr<ICUConverterWrapper> m_cachedConverterICU;
};

#if USE(WEB_THREAD)
PAL_EXPORT ThreadGlobalData& threadGlobalData();
#else
PAL_EXPORT ThreadGlobalData& threadGlobalData() PURE_FUNCTION;
#endif

} // namespace PAL
