/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 4, 2024.
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
#ifndef mach_o_Symbol_h
#define mach_o_Symbol_h

#include <stdint.h>

#include "MachODefines.h"
#include "CString.h"

namespace mach_o {

/*!
 * @class Symbol
 *
 * @abstract
 *      Abstraction for symbols in mach-o final linked executables
 */
class VIS_HIDDEN Symbol
{
public:
    Symbol() = default;
    bool    operator==(const Symbol&) const;
    bool    operator!=(const Symbol& other) const { return !operator==(other); }

    enum class Scope: uint8_t { translationUnit, wasLinkageUnit, linkageUnit, autoHide, global, globalNeverStrip };

    CString      name() const                   { return _name; }
    uint64_t     implOffset() const;            // fails for re-exports and absolute
    Scope        scope() const                  { return _scope; }    // global vs local symbol
    bool         isWeakDef() const              { return _weakDef; }
    bool         dontDeadStrip() const          { return _dontDeadStrip; }
    bool         cold() const                   { return _cold; }
    bool         isThumb() const                { return _isThumb; }
    bool         isThreadLocal() const          { return (_kind == Kind::threadLocal); }
    bool         isDynamicResolver(uint64_t& resolverStubOffset) const;
    bool         isFunctionVariant(uint32_t& functionVariantTableIndex) const;
    bool         isReExport(int& libOrdinal, const char*& importName) const;
    bool         isAbsolute(uint64_t& absAddress) const;
    bool         isUndefined() const;
    bool         isUndefined(int& libOrdinal, bool& weakImport) const;
    bool         isRegular(uint64_t& implOffset) const;
    bool         isThreadLocal(uint64_t& implOffset) const;
    bool         isTentativeDef() const;
    bool         isTentativeDef(uint64_t& size, uint8_t& p2align) const;
    uint8_t      sectionOrdinal() const { return _sectOrdinal; }
    bool         isAltEntry(uint64_t& implOffset) const;
    bool         isAltEntry() const { return _kind == Kind::altEntry; }

    void         setName(const char* n);
    void         setimplOffset(uint64_t);
    void         setDontDeadStrip()         { _dontDeadStrip = true; }
    void         setCold()                  { _cold = true; }
    void         setWeakDef()               { _weakDef = true; }
    void         setNotWeakDef()            { _weakDef = false; }
    void         setIsThumb()               { _isThumb = true; }

    static Symbol makeRegularExport(CString name, uint64_t imageOffset, uint8_t sectNum, bool dontDeadStrip, bool cold, bool neverStrip=false, bool isThumb=false);
    static Symbol makeRegularHidden(CString name, uint64_t imageOffset, uint8_t sectNum, bool dontDeadStrip, bool cold, bool isThumb=false);
    static Symbol makeRegularLocal(CString name, uint64_t imageOffset, uint8_t sectNum, bool dontDeadStrip, bool cold, bool isThumb=false);
    static Symbol makeRegularWasPrivateExtern(CString name, uint64_t imageOffset, uint8_t sectNum, bool dontDeadStrip, bool cold, bool isThumb=false);
    static Symbol makeWeakDefAutoHide(CString name, uint64_t imageOffset, uint8_t sectNum, bool dontDeadStrip, bool cold, bool isThumb=false);   // given the current encoding in mach-o, only weak-defs can be auto-hide
    static Symbol makeWeakDefExport(CString name, uint64_t imageOffset, uint8_t sectNum, bool dontDeadStrip, bool cold, bool isThumb=false);
    static Symbol makeAltEntry(CString name, uint64_t imageOffset, uint8_t sectNum, Scope s, bool dontDeadStrip, bool cold, bool weakDef, bool isThumb=false);
    static Symbol makeWeakDefHidden(CString name, uint64_t imageOffset, uint8_t sectNum, bool dontDeadStrip, bool cold, bool isThumb=false);
    static Symbol makeWeakDefWasPrivateExtern(CString name, uint64_t imageOffset, uint8_t sectNum, bool dontDeadStrip, bool cold, bool isThumb=false);
    static Symbol makeDynamicResolver(CString name, uint8_t sectNum, uint64_t stubImageOffset, uint64_t funcImageOffset, Symbol::Scope=Symbol::Scope::global);
    static Symbol makeFunctionVariantExport(CString name, uint8_t sectNum, uint64_t imageOffsetOfDefault, uint32_t functionVariantTableIndex);
    static Symbol makeThreadLocalExport(CString name, uint64_t imageOffset, uint8_t sectOrd, bool dontDeadStrip, bool cold, bool weakDef);
    static Symbol makeAbsolute(CString name, uint64_t address, bool dontDeadStrip, Scope scope, uint8_t sectNum=0);
    static Symbol makeReExport(CString name, int libOrdinal, const char* importName=nullptr, Symbol::Scope=Symbol::Scope::global);
    static Symbol makeUndefined(CString name, int libOrdinal, bool weakImport=false);
    static Symbol makeTentativeDef(CString name, uint64_t size, uint8_t alignP2, bool dontDeadStrip, bool cold);
    static Symbol makeHiddenTentativeDef(CString name, uint64_t size, uint8_t alignP2, bool dontDeadStrip, bool cold);

private:
    Symbol(CString name) : _name(name) { }
    enum class Kind: uint8_t { regular, altEntry, resolver, absolute, reExport, threadLocal, tentative, undefine, functionVariant };
    CString      _name                  = "";
    uint64_t     _implOffset            = 0;   // resolver => offset to stub, re-exports,undefined => libOrdinal, absolute => address, tentative => size
    union {
        const char* importName;
        uint64_t    resolverStubOffset = 0;
        uint32_t    functionVariantTableIndex;
    } _u;
    Kind         _kind                  = Kind::regular;
    uint8_t      _sectOrdinal           = 0;
    Scope        _scope                 = Scope::translationUnit;  // global vs local
    bool         _weakDef               = false;  // regular only
    bool         _dontDeadStrip         = false;  // regular only
    bool         _cold                  = false;  // regular only
    bool         _weakImport            = false;  // undefines only
    bool         _isThumb               = false;  // regular only
};
static_assert(sizeof(Symbol) == 24+2*sizeof(void*));

} // namespace mach_o

#endif // mach_o_Symbol_h


