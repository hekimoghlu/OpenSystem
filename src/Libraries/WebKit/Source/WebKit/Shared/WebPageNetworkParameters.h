/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 2, 2021.
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

#include <wtf/text/WTFString.h>

namespace IPC {
class Decoder;
class Encoder;
}

namespace WebKit {

class WebPageNetworkParameters {
public:
    WebPageNetworkParameters(const String& attributedBundleIdentifier)
        : m_attributedBundleIdentifier(attributedBundleIdentifier) { }

    WebPageNetworkParameters() = default;
    WebPageNetworkParameters(WTF::HashTableDeletedValueType)
        : m_attributedBundleIdentifier(WTF::HashTableDeletedValue) { }
    bool isHashTableDeletedValue() const { return m_attributedBundleIdentifier.isHashTableDeletedValue(); }
    unsigned hash() const { return m_attributedBundleIdentifier.hash(); }
    friend bool operator==(const WebPageNetworkParameters&, const WebPageNetworkParameters&) = default;

    const String& attributedBundleIdentifier() const { return m_attributedBundleIdentifier; }
private:
    String m_attributedBundleIdentifier;
};

} // namespace WebKit

namespace WTF {

template<> struct DefaultHash<WebKit::WebPageNetworkParameters> {
    static unsigned hash(const WebKit::WebPageNetworkParameters& key) { return key.hash(); }
    static bool equal(const WebKit::WebPageNetworkParameters& a, const WebKit::WebPageNetworkParameters& b) { return a == b; }
    static constexpr bool safeToCompareToEmptyOrDeleted = false;
};

template<> struct HashTraits<WebKit::WebPageNetworkParameters> : public SimpleClassHashTraits<WebKit::WebPageNetworkParameters> { };

} // namespace WTF
