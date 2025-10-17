/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 26, 2023.
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

#include "LazyOperandValueProfile.h"

namespace JSC {

class ScriptExecutable;
class CodeBlock;

class LazyOperandValueProfileParser;

class CompressedLazyValueProfileHolder {
    WTF_MAKE_NONCOPYABLE(CompressedLazyValueProfileHolder);
public:
    CompressedLazyValueProfileHolder() = default;

    void computeUpdatedPredictions(const ConcurrentJSLocker&, CodeBlock*);

    LazyOperandValueProfile* addOperandValueProfile(const LazyOperandValueProfileKey&);
    JSValue* addSpeculationFailureValueProfile(BytecodeIndex);

    UncheckedKeyHashMap<BytecodeIndex, JSValue*> speculationFailureValueProfileBucketsMap();

private:
    friend class LazyOperandValueProfileParser;

    inline void initializeData();

    struct LazyValueProfileHolder {
        WTF_MAKE_STRUCT_TZONE_ALLOCATED(LazyValueProfileHolder);
        ConcurrentVector<LazyOperandValueProfile, 8> operandValueProfiles;
        ConcurrentVector<std::pair<BytecodeIndex, JSValue>, 8> speculationFailureValueProfileBuckets;
    };

    std::unique_ptr<LazyValueProfileHolder> m_data;
};

class LazyOperandValueProfileParser {
    WTF_MAKE_NONCOPYABLE(LazyOperandValueProfileParser);
public:
    LazyOperandValueProfileParser() = default;

    void initialize(CompressedLazyValueProfileHolder&);

    LazyOperandValueProfile* getIfPresent(const LazyOperandValueProfileKey& key) const;

    SpeculatedType prediction(const ConcurrentJSLocker&, const LazyOperandValueProfileKey&) const;
private:
    UncheckedKeyHashMap<LazyOperandValueProfileKey, LazyOperandValueProfile*> m_map;
};

} // namespace JSC
