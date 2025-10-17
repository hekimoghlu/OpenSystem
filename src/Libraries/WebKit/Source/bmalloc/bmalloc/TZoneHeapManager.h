/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 13, 2025.
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

#include "BExport.h"

#if BUSE(TZONE)

#include "Map.h"
#include "Mutex.h"
#include "SegmentedVector.h"
#include "TZoneHeap.h"
#include <CommonCrypto/CommonDigest.h>
#include <mutex>

#if BUSE(LIBPAS)
#include "bmalloc_heap_ref.h"

namespace bmalloc { namespace api {

#define TZONE_VERBOSE_DEBUG 0

extern BEXPORT class TZoneHeapManager* tzoneHeapManager;

class TZoneHeapManager {
    enum class State {
        Uninitialized,
        Seeded,
        StartedRegisteringTypes
    };

    static const unsigned typeNameLen = 12;

    typedef uint64_t SHA256ResultAsUnsigned[CC_SHA256_DIGEST_LENGTH / sizeof(uint64_t)];
    static_assert(!(CC_SHA256_DIGEST_LENGTH % sizeof(uint64_t)));

    struct TZoneBucket {
        bmalloc_type type;
        pas_heap_ref heapref;
        char typeName[typeNameLen];
    };

    struct TZoneTypeBuckets {
        unsigned numberOfBuckets;
#if TZONE_VERBOSE_DEBUG
        unsigned numberOfTypesThisSizeClass;
        unsigned usedBucketBitmap;
        Vector<unsigned> bucketUseCounts;
#endif
        TZoneBucket buckets[1];
    };

// TZoneTypeBuckets already includes room for 1 bucket. Hence, we only need to add count - 1 buckets.
#define SIZE_TZONE_TYPE_BUCKETS(count) (sizeof(struct TZoneTypeBuckets) + (count - 1) * sizeof(TZoneBucket))

    struct TZoneTypeKey {
        TZoneTypeKey() = default;
        TZoneTypeKey(void* address, unsigned size, unsigned alignment)
            : address(address)
            , size(size)
            , alignment(alignment)
        {
            m_key = reinterpret_cast<uintptr_t>(address) << 12 ^ size << 3 ^ alignment >> 3;
        }

        inline unsigned long key() const { return m_key; }

        static unsigned long hash(TZoneTypeKey value)
        {
            return value.m_key;
        }

        bool operator==(const TZoneTypeKey& other) const
        {
            return address == other.address
                && size == other.size
                && alignment == other.alignment;
        }

        bool operator<(const TZoneTypeKey& other) const
        {
            if (address != other.address)
                return address < other.address;

            if (size != other.size)
                return size < other.size;

            return alignment < other.alignment;
        }

        operator bool() const
        {
            return !!key();
        }

        void* address = nullptr;
        unsigned size = 0;
        unsigned alignment = 0;
        uintptr_t m_key = 0;
    };

protected:
    TZoneHeapManager();

public:
    TZoneHeapManager(TZoneHeapManager &other) = delete;
    void operator=(const TZoneHeapManager &) = delete;

    BEXPORT static void requirePerBootSeed();
    BEXPORT static void setBucketParams(unsigned smallSizeCount, unsigned largeSizeCount = 0, unsigned smallSizeLimit = 0);

    BEXPORT static bool isReady();

    BEXPORT static void ensureSingleton();
    BINLINE static TZoneHeapManager& singleton()
    {
        BASSERT(tzoneHeapManager);
        return *tzoneHeapManager;
    }

    static void setHasDisableTZoneEntitlementCallback(bool (*hasDisableTZoneEntitlement)());

    pas_heap_ref* heapRefForTZoneType(const TZoneSpecification&);
    pas_heap_ref* heapRefForTZoneTypeDifferentSize(size_t requestedSize, const TZoneSpecification&);

    BEXPORT void dumpRegisteredTypes();

    enum class AllocationMode {
        TZoneEnabled,
        TZoneDisabled,
    };

    static bool s_tzoneEnabled;
private:
    void init();

    BINLINE Mutex& mutex() { return m_mutex; }
    BINLINE Mutex& differentSizeMutex() { return m_differentSizeMutex; }

    BINLINE pas_heap_ref* heapRefForTZoneType(const TZoneSpecification&, LockHolder&);

    inline static unsigned bucketCountForSizeClass(SizeAndAlignment::Value);

    inline unsigned tzoneBucketForKey(const TZoneSpecification&, unsigned bucketCountForSize, LockHolder&);
    TZoneTypeBuckets* populateBucketsForSizeClass(LockHolder&, SizeAndAlignment::Value);

    static TZoneHeapManager::State s_state;
    Mutex m_mutex;
    Mutex m_differentSizeMutex;
    uint64_t m_tzoneKeySeed;
#if TZONE_VERBOSE_DEBUG
    unsigned largestBucketCount { 0 };
    Vector<SizeAndAlignment::Value> m_typeSizes;
#endif
    Map<SizeAndAlignment::Value, TZoneTypeBuckets*, SizeAndAlignment> m_heapRefsBySizeAndAlignment;
    Map<TZoneTypeKey, pas_heap_ref*, TZoneTypeKey> m_differentSizedHeapRefs;
};

} } // namespace bmalloc::api

#endif // BUSE(LIBPAS)

#endif // BUSE(TZONE)
