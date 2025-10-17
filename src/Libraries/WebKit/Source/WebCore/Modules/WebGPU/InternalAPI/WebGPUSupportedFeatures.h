/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 21, 2023.
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
#include <wtf/Vector.h>
#include <wtf/text/WTFString.h>

namespace WebCore::WebGPU {

class SupportedFeatures final : public RefCounted<SupportedFeatures> {
public:
    static Ref<SupportedFeatures> create(Vector<String>&& features)
    {
        return adoptRef(*new SupportedFeatures(WTFMove(features)));
    }

    static Ref<SupportedFeatures> create(const Vector<String>& features)
    {
        return adoptRef(*new SupportedFeatures(features));
    }

    static Ref<SupportedFeatures> clone(const SupportedFeatures& features)
    {
        return adoptRef(*new SupportedFeatures(Vector<String>(features.features())));
    }

    const Vector<String>& features() const { return m_features; }

private:
    SupportedFeatures(Vector<String>&& features)
        : m_features(WTFMove(features))
    {
    }

    SupportedFeatures(const Vector<String>& features)
        : m_features(features)
    {
    }

    SupportedFeatures(const SupportedFeatures&) = delete;
    SupportedFeatures(SupportedFeatures&&) = delete;
    SupportedFeatures& operator=(const SupportedFeatures&) = delete;
    SupportedFeatures& operator=(SupportedFeatures&&) = delete;

    Vector<String> m_features;
};

} // namespace WebCore::WebGPU
