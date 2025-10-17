/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 9, 2024.
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
#include "ASLRTracker.h"

using namespace cache_builder;

void ASLR_Tracker::setDataRegion(const void* rwRegionStart, size_t rwRegionSize)
{
#if BUILDING_APP_CACHE_UTIL
    assert((rwRegionSize % kMinimumFixupAlignment) == 0);
#else
    // Its ok for the rwRegion size to not be a multiple of the minimum alignment
    // However, that implies that we can't have a pointer on the end of the buffer
    // Eg, we could have a 12 byte buffer with an 8-byte pointer starting at an offset of 8
    // We'll shrink the size to account for this as we couldn't have the pointer at the end anyway
    size_t remainder = (rwRegionSize % kMinimumFixupAlignment);
    if ( remainder != 0 ) {
        rwRegionSize -= remainder;
    }
#endif

    _regionStart = (uint8_t*)rwRegionStart;
    _regionEnd   = (uint8_t*)rwRegionStart + rwRegionSize;
    _bitmap.resize(rwRegionSize / kMinimumFixupAlignment);

#if BUILDING_APP_CACHE_UTIL
    _cacheLevels.resize(rwRegionSize / kMinimumFixupAlignment, (uint8_t)~0U);
#endif
}

void ASLR_Tracker::add(void* loc, uint8_t level)
{
    if (!_enabled)
        return;
    uint8_t* p = (uint8_t*)loc;
    assert(p >= _regionStart);
    assert(p < _regionEnd);
    _bitmap[(p-_regionStart)/kMinimumFixupAlignment] = true;

#if BUILDING_APP_CACHE_UTIL
    if ( level != (uint8_t)~0U ) {
        _cacheLevels[(p-_regionStart)/kMinimumFixupAlignment] = level;
    }
#endif
}

void ASLR_Tracker::remove(void* loc)
{
    if (!_enabled)
        return;
    uint8_t* p = (uint8_t*)loc;
    assert(p >= _regionStart);
    assert(p < _regionEnd);
    _bitmap[(p-_regionStart)/kMinimumFixupAlignment] = false;
}

bool ASLR_Tracker::has(void* loc, uint8_t* level) const
{
    if (!_enabled)
        return true;
    uint8_t* p = (uint8_t*)loc;
    assert(p >= _regionStart);
    assert(p < _regionEnd);

    if ( _bitmap[(p-_regionStart)/kMinimumFixupAlignment] ) {
#if BUILDING_APP_CACHE_UTIL
        if ( level != nullptr ) {
            uint8_t levelValue = _cacheLevels[(p-_regionStart)/kMinimumFixupAlignment];
            if ( levelValue != (uint8_t)~0U )
                *level = levelValue;
        }
#endif
        return true;
    }
    return false;
}

void ASLR_Tracker::setRebaseTarget32(void*p, uint32_t targetVMAddr)
{
    _rebaseTarget32[p] = targetVMAddr;
}

void ASLR_Tracker::setRebaseTarget64(void*p, uint64_t targetVMAddr)
{
    _rebaseTarget64[p] = targetVMAddr;
}

bool ASLR_Tracker::hasRebaseTarget32(void* p, uint32_t* vmAddr) const
{
    auto pos = _rebaseTarget32.find(p);
    if ( pos == _rebaseTarget32.end() )
        return false;
    *vmAddr = pos->second;
    return true;
}

bool ASLR_Tracker::hasRebaseTarget64(void* p, uint64_t* vmAddr) const
{
    auto pos = _rebaseTarget64.find(p);
    if ( pos == _rebaseTarget64.end() )
        return false;
    *vmAddr = pos->second;
    return true;
}

void ASLR_Tracker::forEachFixup(void (^callback)(void* loc, bool& stop))
{
    for ( size_t i = 0, e = this->_bitmap.size(); i != e; ++i ) {
        if ( !_bitmap[i] )
            continue;

        uint64_t offset = i * kMinimumFixupAlignment;

        bool stop = false;
        callback(this->_regionStart + offset, stop);
        if ( stop )
            break;
    }
}

#if BUILDING_APP_CACHE_UTIL
void ASLR_Tracker::setHigh8(void* p, uint8_t high8)
{
    _high8Map[p] = high8;
}

void ASLR_Tracker::setAuthData(void* p, uint16_t diversity, bool hasAddrDiv, uint8_t key)
{
    _authDataMap[p] = {diversity, hasAddrDiv, key};
}

bool ASLR_Tracker::hasHigh8(void* p, uint8_t* highByte) const
{
    auto pos = _high8Map.find(p);
    if ( pos == _high8Map.end() )
        return false;
    *highByte = pos->second;
    return true;
}

bool ASLR_Tracker::hasAuthData(void* p, uint16_t* diversity, bool* hasAddrDiv, uint8_t* key) const
{
    auto pos = _authDataMap.find(p);
    if ( pos == _authDataMap.end() )
        return false;
    *diversity  = pos->second.diversity;
    *hasAddrDiv = pos->second.addrDiv;
    *key        = pos->second.key;
    return true;
}

std::vector<void*> ASLR_Tracker::getRebaseTargets() const {
    std::vector<void*> targets;
    for (const auto& target : _rebaseTarget32)
        targets.push_back(target.first);
    for (const auto& target : _rebaseTarget64)
        targets.push_back(target.first);
    return targets;
}
#endif

#if BUILDING_CACHE_BUILDER || BUILDING_CACHE_BUILDER_UNIT_TESTS
void ASLR_Tracker::clearRebaseTargetsMaps()
{
    this->_rebaseTarget32.clear();
    this->_rebaseTarget64.clear();
}
#endif
