/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 28, 2022.
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

#include "MacroAssemblerCodeRef.h"

namespace JSC {

enum class NearCallMode : uint8_t { Regular, Tail };

template<PtrTag> class CodeLocationInstruction;
template<PtrTag> class CodeLocationLabel;
template<PtrTag> class CodeLocationJump;
template<PtrTag> class CodeLocationCall;
template<PtrTag> class CodeLocationNearCall;
template<PtrTag> class CodeLocationDataLabelCompact;
template<PtrTag> class CodeLocationDataLabel32;
template<PtrTag> class CodeLocationDataLabelPtr;
template<PtrTag> class CodeLocationConvertibleLoad;

// The CodeLocation* types are all pretty much do-nothing wrappers around
// CodePtr. These classes only exist to provide type-safety when linking and
// patching code.
//
// The one new piece of functionallity introduced by these classes is the
// ability to create (or put another way, to re-discover) another CodeLocation
// at an offset from one you already know.  When patching code to optimize it
// we often want to patch a number of instructions that are short, fixed
// offsets apart.  To reduce memory overhead we will only retain a pointer to
// one of the instructions, and we will use the *AtOffset methods provided by
// CodeLocationCommon to find the other points in the code to modify.
template<PtrTag tag>
class CodeLocationCommon : public CodePtr<tag> {
    using Base = CodePtr<tag>;
public:
    template<PtrTag resultTag = tag> CodeLocationInstruction<resultTag> instructionAtOffset(int offset);
    template<PtrTag resultTag = tag> CodeLocationLabel<resultTag> labelAtOffset(int offset);
    template<PtrTag resultTag = tag> CodeLocationJump<resultTag> jumpAtOffset(int offset);
    template<PtrTag resultTag = tag> CodeLocationCall<resultTag> callAtOffset(int offset);
    template<PtrTag resultTag = tag> CodeLocationNearCall<resultTag> nearCallAtOffset(int offset, NearCallMode);
    template<PtrTag resultTag = tag> CodeLocationDataLabelPtr<resultTag> dataLabelPtrAtOffset(int offset);
    template<PtrTag resultTag = tag> CodeLocationDataLabel32<resultTag> dataLabel32AtOffset(int offset);
    template<PtrTag resultTag = tag> CodeLocationDataLabelCompact<resultTag> dataLabelCompactAtOffset(int offset);
    template<PtrTag resultTag = tag> CodeLocationConvertibleLoad<resultTag> convertibleLoadAtOffset(int offset);

    template<typename T = void*>
    T dataLocation() const { return Base::template dataLocation<T>(); }

    template<typename T, typename = std::enable_if_t<std::is_base_of<CodeLocationCommon<tag>, T>::value>>
    operator T()
    {
        return T(CodePtr<tag>::fromTaggedPtr(this->taggedPtr()));
    }

protected:
    CodeLocationCommon()
    {
    }

    CodeLocationCommon(CodePtr<tag> location)
        : CodePtr<tag>(location)
    {
    }
};

template<PtrTag tag>
class CodeLocationInstruction : public CodeLocationCommon<tag> {
public:
    CodeLocationInstruction() { }
    explicit CodeLocationInstruction(CodePtr<tag> location)
        : CodeLocationCommon<tag>(location) { }
    explicit CodeLocationInstruction(void* location)
        : CodeLocationCommon<tag>(CodePtr<tag>(location)) { }
};

template<PtrTag tag>
class CodeLocationLabel : public CodeLocationCommon<tag> {
public:
    CodeLocationLabel() { }
    explicit CodeLocationLabel(CodePtr<tag> location)
        : CodeLocationCommon<tag>(location) { }
    explicit CodeLocationLabel(void* location)
        : CodeLocationCommon<tag>(CodePtr<tag>(location)) { }

    template<PtrTag newTag>
    CodeLocationLabel<newTag> retagged() { return CodeLocationLabel<newTag>(CodeLocationCommon<tag>::template retagged<newTag>()); }

    template<typename T = void*>
    T untaggedPtr() const { return CodeLocationCommon<tag>::template untaggedPtr<T>(); }

    template<typename T = void*>
    T dataLocation() const { return CodeLocationCommon<tag>::template dataLocation<T>(); }
};

template<PtrTag tag>
class CodeLocationJump : public CodeLocationCommon<tag> {
public:
    CodeLocationJump() { }
    explicit CodeLocationJump(CodePtr<tag> location)
        : CodeLocationCommon<tag>(location) { }
    explicit CodeLocationJump(void* location)
        : CodeLocationCommon<tag>(CodePtr<tag>(location)) { }

    template<PtrTag newTag>
    CodeLocationJump<newTag> retagged() { return CodeLocationJump<newTag>(CodePtr<tag>::template retagged<newTag>()); }
};

template<PtrTag tag>
class CodeLocationCall : public CodeLocationCommon<tag> {
public:
    CodeLocationCall() { }
    explicit CodeLocationCall(CodePtr<tag> location)
        : CodeLocationCommon<tag>(location) { }
    explicit CodeLocationCall(void* location)
        : CodeLocationCommon<tag>(CodePtr<tag>(location)) { }

    template<PtrTag newTag>
    CodeLocationCall<newTag> retagged() { return CodeLocationCall<newTag>(CodeLocationCommon<tag>::template retagged<newTag>()); }
};

template<PtrTag tag>
class CodeLocationNearCall : public CodeLocationCommon<tag> {
public:
    CodeLocationNearCall() { }
    explicit CodeLocationNearCall(CodePtr<tag> location, NearCallMode callMode)
        : CodeLocationCommon<tag>(location), m_callMode(callMode) { }
    explicit CodeLocationNearCall(void* location, NearCallMode callMode)
        : CodeLocationCommon<tag>(CodePtr<tag>(location)), m_callMode(callMode) { }
    NearCallMode callMode() { return m_callMode; }
private:
    NearCallMode m_callMode { NearCallMode::Regular };
};

template<PtrTag tag>
class CodeLocationDataLabel32 : public CodeLocationCommon<tag> {
public:
    CodeLocationDataLabel32() { }
    explicit CodeLocationDataLabel32(CodePtr<tag> location)
        : CodeLocationCommon<tag>(location) { }
    explicit CodeLocationDataLabel32(void* location)
        : CodeLocationCommon<tag>(CodePtr<tag>(location)) { }
};

template<PtrTag tag>
class CodeLocationDataLabelCompact : public CodeLocationCommon<tag> {
public:
    CodeLocationDataLabelCompact() { }
    explicit CodeLocationDataLabelCompact(CodePtr<tag> location)
        : CodeLocationCommon<tag>(location) { }
    explicit CodeLocationDataLabelCompact(void* location)
        : CodeLocationCommon<tag>(CodePtr<tag>(location)) { }
};

template<PtrTag tag>
class CodeLocationDataLabelPtr : public CodeLocationCommon<tag> {
public:
    CodeLocationDataLabelPtr() { }
    explicit CodeLocationDataLabelPtr(CodePtr<tag> location)
        : CodeLocationCommon<tag>(location) { }
    explicit CodeLocationDataLabelPtr(void* location)
        : CodeLocationCommon<tag>(CodePtr<tag>(location)) { }
};

template<PtrTag tag>
class CodeLocationConvertibleLoad : public CodeLocationCommon<tag> {
public:
    CodeLocationConvertibleLoad() { }
    explicit CodeLocationConvertibleLoad(CodePtr<tag> location)
        : CodeLocationCommon<tag>(location) { }
    explicit CodeLocationConvertibleLoad(void* location)
        : CodeLocationCommon<tag>(CodePtr<tag>(location)) { }
};

template<PtrTag tag>
template<PtrTag resultTag>
inline CodeLocationInstruction<resultTag> CodeLocationCommon<tag>::instructionAtOffset(int offset)
{
    ASSERT_VALID_CODE_OFFSET(offset);
    return CodeLocationInstruction<resultTag>(tagCodePtr<resultTag>(dataLocation<char*>() + offset));
}

template<PtrTag tag>
template<PtrTag resultTag>
inline CodeLocationLabel<resultTag> CodeLocationCommon<tag>::labelAtOffset(int offset)
{
    ASSERT_VALID_CODE_OFFSET(offset);
    return CodeLocationLabel<resultTag>(tagCodePtr<resultTag>(dataLocation<char*>() + offset));
}

template<PtrTag tag>
template<PtrTag resultTag>
inline CodeLocationJump<resultTag> CodeLocationCommon<tag>::jumpAtOffset(int offset)
{
    ASSERT_VALID_CODE_OFFSET(offset);
    return CodeLocationJump<resultTag>(tagCodePtr<resultTag>(dataLocation<char*>() + offset));
}

template<PtrTag tag>
template<PtrTag resultTag>
inline CodeLocationCall<resultTag> CodeLocationCommon<tag>::callAtOffset(int offset)
{
    ASSERT_VALID_CODE_OFFSET(offset);
    return CodeLocationCall<resultTag>(tagCodePtr<resultTag>(dataLocation<char*>() + offset));
}

template<PtrTag tag>
template<PtrTag resultTag>
inline CodeLocationNearCall<resultTag> CodeLocationCommon<tag>::nearCallAtOffset(int offset, NearCallMode callMode)
{
    ASSERT_VALID_CODE_OFFSET(offset);
    return CodeLocationNearCall<resultTag>(tagCodePtr<resultTag>(dataLocation<char*>() + offset), callMode);
}

template<PtrTag tag>
template<PtrTag resultTag>
inline CodeLocationDataLabelPtr<resultTag> CodeLocationCommon<tag>::dataLabelPtrAtOffset(int offset)
{
    ASSERT_VALID_CODE_OFFSET(offset);
    return CodeLocationDataLabelPtr<resultTag>(tagCodePtr<resultTag>(dataLocation<char*>() + offset));
}

template<PtrTag tag>
template<PtrTag resultTag>
inline CodeLocationDataLabel32<resultTag> CodeLocationCommon<tag>::dataLabel32AtOffset(int offset)
{
    ASSERT_VALID_CODE_OFFSET(offset);
    return CodeLocationDataLabel32<resultTag>(tagCodePtr<resultTag>(dataLocation<char*>() + offset));
}

template<PtrTag tag>
template<PtrTag resultTag>
inline CodeLocationDataLabelCompact<resultTag> CodeLocationCommon<tag>::dataLabelCompactAtOffset(int offset)
{
    ASSERT_VALID_CODE_OFFSET(offset);
    return CodeLocationDataLabelCompact<resultTag>(tagCodePtr<resultTag>(dataLocation<char*>() + offset));
}

template<PtrTag tag>
template<PtrTag resultTag>
inline CodeLocationConvertibleLoad<resultTag> CodeLocationCommon<tag>::convertibleLoadAtOffset(int offset)
{
    ASSERT_VALID_CODE_OFFSET(offset);
    return CodeLocationConvertibleLoad<resultTag>(tagCodePtr<resultTag>(dataLocation<char*>() + offset));
}

} // namespace JSC
