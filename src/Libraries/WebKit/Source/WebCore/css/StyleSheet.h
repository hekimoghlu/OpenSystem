/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 7, 2023.
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

#include "CSSParserMode.h"
#include <wtf/Forward.h>
#include <wtf/RefCounted.h>

namespace WTF {
class TextStream;
}

namespace WebCore {

class CSSImportRule;
class MediaList;
class Node;
class StyleSheet;

class StyleSheet : public RefCounted<StyleSheet> {
public:
    virtual ~StyleSheet();

    virtual bool disabled() const = 0;
    virtual void setDisabled(bool) = 0;
    virtual Node* ownerNode() const = 0;
    virtual StyleSheet* parentStyleSheet() const { return 0; }
    virtual String href() const = 0;
    virtual String title() const = 0;
    virtual MediaList* media() const { return 0; }
    virtual String type() const = 0;

    virtual CSSImportRule* ownerRule() const { return 0; }
    virtual void clearOwnerNode() = 0;
    virtual URL baseURL() const = 0;
    virtual bool isLoading() const = 0;
    virtual bool isCSSStyleSheet() const { return false; }
    virtual bool isXSLStyleSheet() const { return false; }

    virtual String debugDescription() const = 0;
};

TextStream& operator<<(TextStream&, const StyleSheet&);

} // namespace WebCore
