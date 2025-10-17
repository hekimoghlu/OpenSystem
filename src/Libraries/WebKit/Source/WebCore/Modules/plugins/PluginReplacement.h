/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 24, 2024.
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

#include "RenderPtr.h"
#include <wtf/text/WTFString.h>

namespace WebCore {

class HTMLPlugInElement;
class RenderElement;
class RenderStyle;
class RenderTreePosition;
class ShadowRoot;

class PluginReplacement : public RefCounted<PluginReplacement> {
public:
    virtual ~PluginReplacement() = default;

    virtual void installReplacement(ShadowRoot&) = 0;

    virtual bool willCreateRenderer() { return false; }
    virtual RenderPtr<RenderElement> createElementRenderer(HTMLPlugInElement&, RenderStyle&&, const RenderTreePosition&) = 0;
};

typedef Ref<PluginReplacement> (*CreatePluginReplacement)(HTMLPlugInElement&, const Vector<AtomString>& paramNames, const Vector<AtomString>& paramValues);
typedef bool (*PluginReplacementSupportsType)(const String&);
typedef bool (*PluginReplacementSupportsFileExtension)(StringView);
typedef bool (*PluginReplacementSupportsURL)(const URL&);

class ReplacementPlugin {
public:
    ReplacementPlugin(CreatePluginReplacement constructor, PluginReplacementSupportsType supportsType, PluginReplacementSupportsFileExtension supportsFileExtension, PluginReplacementSupportsURL supportsURL)
        : m_constructor(constructor)
        , m_supportsType(supportsType)
        , m_supportsFileExtension(supportsFileExtension)
        , m_supportsURL(supportsURL)
    {
    }

    explicit ReplacementPlugin(const ReplacementPlugin& other)
        : m_constructor(other.m_constructor)
        , m_supportsType(other.m_supportsType)
        , m_supportsFileExtension(other.m_supportsFileExtension)
        , m_supportsURL(other.m_supportsURL)
    {
    }

    Ref<PluginReplacement> create(HTMLPlugInElement& element, const Vector<AtomString>& paramNames, const Vector<AtomString>& paramValues) const { return m_constructor(element, paramNames, paramValues); }
    bool supportsType(const String& mimeType) const { return m_supportsType(mimeType); }
    bool supportsFileExtension(StringView extension) const { return m_supportsFileExtension(extension); }
    bool supportsURL(const URL& url) const { return m_supportsURL(url); }

private:
    CreatePluginReplacement m_constructor;
    PluginReplacementSupportsType m_supportsType;
    PluginReplacementSupportsFileExtension m_supportsFileExtension;
    PluginReplacementSupportsURL m_supportsURL;
};

typedef void (*PluginReplacementRegistrar)(const ReplacementPlugin&);

}
