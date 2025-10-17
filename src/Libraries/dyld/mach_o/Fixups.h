/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 11, 2024.
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
#ifndef mach_o_Fixups_h
#define mach_o_Fixups_h

#include "CString.h"

#include "MachODefines.h"

namespace mach_o {



struct MappedSegment
{
    uint64_t            runtimeOffset;
    uint64_t            runtimeSize;
    uint64_t            fileOffset;
    void*               content;
    std::string_view    segName;
    bool                readable;
    bool                writable;
    bool                executable;
};



/*!
 * @class Fixup
 *
 * @abstract
 *      Class for encapsulating everything about a fixup.
 *
 *
 */
struct VIS_HIDDEN Fixup
{
    const void*             location;
    const MappedSegment*    segment;
    bool                    authenticated = false;
    struct {
        uint8_t         key               :  2 = 0,
                        usesAddrDiversity :  1 = 0;
        uint16_t        diversity              = 0;
    }  auth;
    bool                    isBind;
    bool                    isLazyBind; 
    union {
        struct {
            uint32_t    bindOrdinal;   // index into BindTarget array
            int32_t     embeddedAddend;
        } bind;
        struct {
            uint64_t    targetVmOffset; // includes high8
        } rebase;
    };

    bool            operator==(const Fixup& other) const;
    bool            operator!=(const Fixup& other) const { return !this->operator==(other); }
    bool            operator<(const Fixup& other) const  { return (this->location < other.location); }
    const char*     keyName() const;

    struct BindTarget
    {
        CString     symbolName;
        int         libOrdinal  = 0;
        bool        weakImport  = false;
        int64_t     addend      = 0;
    };

    // constructor for a non-auth bind
    Fixup(const void* loc, const MappedSegment* seg, uint32_t bindOrdinal, int32_t embeddedAddend, bool lazy=false)
        : location(loc), segment(seg), isBind(true), isLazyBind(lazy)
    {
        bind.bindOrdinal    = bindOrdinal;
        bind.embeddedAddend = embeddedAddend;
    }

    // constructor for a non-auth rebase
    Fixup(const void* loc, const MappedSegment* seg, uint64_t targetVmOffset)
        : location(loc), segment(seg), isBind(false), isLazyBind(false)
    {
        rebase.targetVmOffset = targetVmOffset;
    }

    // constructor for an auth bind
    Fixup(const void* loc, const MappedSegment* seg, uint32_t bindOrdinal, int32_t embeddedAddend, uint8_t key, bool usesAD, uint16_t div)
        : location(loc), segment(seg), isBind(true), isLazyBind(false)
    {
        bind.bindOrdinal           = bindOrdinal;
        bind.embeddedAddend        = embeddedAddend;
        authenticated              = true;
        auth.key                   = key;
        auth.usesAddrDiversity     = usesAD;
        auth.diversity             = div;
    }

    // constructor for an auth rebase
    Fixup(const void* loc, const MappedSegment* seg, uint64_t targetVmOffset, uint8_t key, bool usesAD, uint16_t div)
        : location(loc), segment(seg), isBind(false), isLazyBind(false)
    {
        rebase.targetVmOffset   = targetVmOffset;
        authenticated           = true;
        auth.key                = key;
        auth.usesAddrDiversity  = usesAD;
        auth.diversity          = div;
    }

    static const char*     keyName(uint8_t keyNum);

};




} // namespace mach_o

#endif /* mach_o_Fixups_h */
