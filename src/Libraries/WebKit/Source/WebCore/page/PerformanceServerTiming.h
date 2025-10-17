/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 16, 2023.
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

#include <wtf/RefCounted.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

class PerformanceServerTiming : public RefCounted<PerformanceServerTiming> {
public:
    static Ref<PerformanceServerTiming> create(String&& name, double duration, String&& description);
    ~PerformanceServerTiming();

    const String& name() const { return m_name; }
    double duration() const { return m_duration; }
    const String& description() const { return m_description; }

private:
    PerformanceServerTiming(String&& name, double duration, String&& description);
    String m_name;
    double m_duration;
    String m_description;
};

} // namespace WebCore
