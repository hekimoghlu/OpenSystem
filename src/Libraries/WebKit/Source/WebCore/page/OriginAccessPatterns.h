/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 1, 2022.
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

#include <wtf/Forward.h>

namespace WebCore {

class UserContentURLPattern;

class OriginAccessPatterns {
public:
    virtual bool anyPatternMatches(const URL&) const = 0;
    virtual ~OriginAccessPatterns() { }
};

class WEBCORE_EXPORT OriginAccessPatternsForWebProcess final : public OriginAccessPatterns {
public:
    static OriginAccessPatternsForWebProcess& singleton();
    void allowAccessTo(const UserContentURLPattern&);
private:
    bool anyPatternMatches(const URL&) const final;
};

class WEBCORE_EXPORT EmptyOriginAccessPatterns final : public OriginAccessPatterns {
public:
    static const EmptyOriginAccessPatterns& singleton();
private:
    bool anyPatternMatches(const URL&) const final;
};

const OriginAccessPatterns& originAccessPatternsForWebProcessOrEmpty();

}
