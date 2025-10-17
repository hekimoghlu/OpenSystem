/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 11, 2022.
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
#ifndef MetadataVisitor_hpp
#define MetadataVisitor_hpp

#include "Defines.h"
#include "MachOFile.h"
#include "Types.h"

#include <optional>

#if BUILDING_CACHE_BUILDER || BUILDING_CACHE_BUILDER_UNIT_TESTS
#include <vector>
#endif

// dyld_closure_util and dyld_info don't slide the pointers, so this tells us if pointers are
// live or in their on-disk representation
// In dyld_shared_cache_util, we make sure to only call the visitors on on-disk caches.
// That is a requirement anyway, as otherwise we'll be trying to chase pointers in in-use objc
// classes
#define POINTERS_ARE_UNSLID (BUILDING_DYLDINFO || BUILDING_CLOSURE_UTIL || BUILDING_SHARED_CACHE_UTIL)

#if POINTERS_ARE_UNSLID
class DyldSharedCache;
#endif

namespace dyld3
{
#if SUPPORT_VM_LAYOUT
struct MachOAnalyzer;
#endif

struct MachOFile;
}

namespace metadata_visitor
{

#if !SUPPORT_VM_LAYOUT
struct Segment
{
    VMAddress               startVMAddr;
    VMAddress               endVMAddr;
    uint8_t*                bufferStart     = nullptr;
    uint32_t                segIndex        = ~0U;

    // When walking the objc in an on-disk binary, we might need the chain format to crack bits
    // of the pointer values.  In cache dylibs this will not be set.  In on-disk binaries with
    // rebase opcodes, this will be set, but to 0, which means not using chains
    std::optional<uint16_t> onDiskDylibChainedPointerFormat;
};
#endif

struct ResolvedValue
{
    void*       value() const;
    VMAddress   vmAddress() const;

#if SUPPORT_VM_LAYOUT
    ResolvedValue(const void* targetValue, VMAddress vmAddr);
#else
    ResolvedValue(const Segment& cacheSegment, VMOffset segmentVMOffset);
    ResolvedValue(const ResolvedValue& parentValue, const void* childLocation);

    std::optional<uint16_t> chainedPointerFormat() const;
    uint32_t segmentIndex() const;
#endif

private:

#if SUPPORT_VM_LAYOUT
    const void* targetValue;
    VMAddress   vmAddr;
#else
    const Segment& cacheSegment;
    VMOffset segmentVMOffset;
#endif
};

struct Visitor
{
    Visitor() = delete;
    Visitor(const Visitor&) = delete;
    Visitor& operator=(const Visitor&) = delete;
    Visitor(Visitor&&) = default;
    Visitor& operator=(Visitor&) = delete;

#if POINTERS_ARE_UNSLID
    // Everying other than the cache builder has MachOAnalyzer available
    Visitor(const DyldSharedCache* dyldCache, const dyld3::MachOAnalyzer* dylibMA,
            std::optional<VMAddress> selectorStringsBaseAddress);
#elif SUPPORT_VM_LAYOUT
    // Everying other than the cache builder has MachOAnalyzer available
    Visitor(const dyld3::MachOAnalyzer* dylibMA);
#else
    // Cache builder dylib
    Visitor(CacheVMAddress cacheBaseAddress, const dyld3::MachOFile* dylibMF,
            std::vector<Segment>&& segments, std::optional<VMAddress> selectorStringsBaseAddress,
            std::vector<uint64_t>&& bindTargets);

    // On disk dylib/executable
    Visitor(VMAddress chainedPointerBaseAddress, const dyld3::MachOFile* dylibMF,
            std::vector<Segment>&& segments, std::optional<VMAddress> selectorStringsBaseAddress,
            std::vector<uint64_t>&& bindTargets);
#endif

    // These helper methods are to aid in pointer chasing.  We start from methods such as forEachClass
    // which find sections in the mach-o, and give us our initial ResolvedValue.  Eg, a ResolvedValue which
    // points to the objc_classlist.  This first value acts as a parent from which we can dereference it to
    // get to the next value, and so on.  Eg, deference the classlist entry to get a Class.  Then when in
    // the Class, its location is the new "parent", from which we can dereference it to find any of the class fields.

    // We have a "parent" ResolvedValue, and the location of a field in the struct it represents.  Returns
    // a new ResolvedValue which points to that field.  Doesn't do any kind of dereferencing, ie, this isn't
    // equivalent to following pointers, but more like returning the address of some field 'x' in "struct { ... x }"
    ResolvedValue getField(const ResolvedValue& parent, const void* fieldPos) const;

    // Dererences the given value.  If it resolves to nullptr, then returns { }, and otherwise returns the new
    // value of the target location
    std::optional<ResolvedValue>    resolveOptionalRebase(const ResolvedValue& value) const;

    // As above, but returns the VMAddress pointed to by this rebase
    std::optional<VMAddress>        resolveOptionalRebaseToVMAddress(const ResolvedValue& value) const;

    // Dereferences the given value.  It is required to be a rebase.  It must not resolve to nullptr
    ResolvedValue                   resolveRebase(const ResolvedValue& value) const;

    // Dereferences the given value.  It may be either a bind or a rebase.  It must not resolve to nullptr
    ResolvedValue                   resolveBindOrRebase(const ResolvedValue& value, bool& wasBind) const;

    // Finds the given VM address in the memory tracked by the visitor, and returns a ResolvedValue which points to it
    ResolvedValue                   getValueFor(VMAddress vmAddr) const;

#if BUILDING_CACHE_BUILDER || BUILDING_CACHE_BUILDER_UNIT_TESTS
    // The value here points to a pointer field in a struct.  This set the location that rebase points to
    void                            updateTargetVMAddress(ResolvedValue& value, CacheVMAddress vmAddr) const;
    void                            setTargetVMAddress(ResolvedValue& value, CacheVMAddress vmAddr,
                                                       const dyld3::MachOFile::PointerMetaData& PMD) const;
#endif

    uint32_t                    pointerSize = 0;

#if BUILDING_CACHE_BUILDER || BUILDING_CACHE_BUILDER_UNIT_TESTS || POINTERS_ARE_UNSLID
    VMAddress                   sharedCacheSelectorStringsBaseAddress() const;
#endif

#if BUILDING_CACHE_BUILDER || BUILDING_CACHE_BUILDER_UNIT_TESTS || POINTERS_ARE_UNSLID
    VMAddress                   getOnDiskDylibChainedPointerBaseAddress() const;
    const dyld3::MachOFile*     mf() const;
    const mach_o::Header*       hdr() const;
    bool                        isOnDiskBinary() const;
#endif


#if BUILDING_CACHE_BUILDER_UNIT_TESTS
    // We need everything public to write tests
public:
#else
protected:
#endif

#if SUPPORT_VM_LAYOUT
    const dyld3::MachOAnalyzer* dylibMA = nullptr;
    VMAddress                   dylibBaseAddress;
#else
    // Is this a cache builder dylib, in the process of being built, or an on-disk dylib/executable
    bool                        isOnDiskDylib = false;
    const dyld3::MachOFile*     dylibMF = nullptr;

    // For an on-disk binary, this is the base address to add to fixup chains
    VMAddress                   onDiskDylibChainedPointerBaseAddress;

    // For a cache binary, this is the base address of the shared cache, to be added to any
    // VMOffsets
    CacheVMAddress              sharedCacheBaseAddress;

    std::vector<Segment>        segments;
    std::vector<uint64_t>       bindTargets;
    std::optional<VMAddress>    selectorStringsBaseAddress;
#endif

#if POINTERS_ARE_UNSLID
    // For an on-disk binary, this is the base address to add to fixup chains
    VMAddress                   onDiskDylibChainedPointerBaseAddress;
    uint16_t                    chainedPointerFormat = 0;
    std::optional<VMAddress>    selectorStringsBaseAddress;

    // If analyzing a shared cache dylib, we might need to crack the shared cache chained fixups
    enum class SharedCacheFormat : uint8_t {
        none            = 0,
        v1              = 1,
        v2_x86_64_tbi   = 2,
        v3              = 3,
        v4              = 4,
        v5              = 5,
    };
    SharedCacheFormat sharedCacheChainedPointerFormat   = SharedCacheFormat::none;
#endif
};

} // namespace metadata_visitor

#endif // MetadataVisitor_h
