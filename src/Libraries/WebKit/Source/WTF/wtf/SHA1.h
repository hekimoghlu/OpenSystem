/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 27, 2025.
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

#include <array>
#include <span>
#include <wtf/text/CString.h>

#if PLATFORM(COCOA)
#include <CommonCrypto/CommonDigest.h>
#endif

#ifdef __OBJC__
#include <objc/objc.h>
#endif

#if USE(CF)
typedef const struct __CFString * CFStringRef;
#endif

// On Cocoa platforms, CoreUtils.h has a SHA1() macro that sometimes get included above here.
#ifdef SHA1
#undef SHA1
#endif

namespace WTF {

class SHA1 {
    WTF_MAKE_FAST_ALLOCATED;
public:
    WTF_EXPORT_PRIVATE SHA1();

    WTF_EXPORT_PRIVATE void addBytes(std::span<const std::byte>);

    void addBytes(std::span<const uint8_t> input)
    {
        addBytes(std::as_bytes(input));
    }

    void addBytes(const CString& input)
    {
        addBytes(input.span());
    }

    WTF_EXPORT_PRIVATE void addUTF8Bytes(StringView);

#if USE(CF)
    WTF_EXPORT_PRIVATE void addUTF8Bytes(CFStringRef);
#ifdef __OBJC__
    void addUTF8Bytes(NSString *string) { addUTF8Bytes((__bridge CFStringRef)string); }
#endif
#endif

    // Size of the SHA1 hash
    WTF_EXPORT_PRIVATE static constexpr size_t hashSize = 20;

    // type for computing SHA1 hash
    typedef std::array<uint8_t, hashSize> Digest;

    WTF_EXPORT_PRIVATE void computeHash(Digest&);

    // Get a hex hash from the digest.
    WTF_EXPORT_PRIVATE static CString hexDigest(const Digest&);

    // Compute the hex digest directly.
    WTF_EXPORT_PRIVATE CString computeHexDigest();

private:
#if PLATFORM(COCOA)
    CC_SHA1_CTX m_context;
#else
    void finalize();
    void processBlock();
    void reset();

    std::array<uint8_t, 64> m_buffer;
    size_t m_cursor; // Number of bytes filled in m_buffer (0-64).
    uint64_t m_totalBytes; // Number of bytes added so far.
    std::array<uint32_t, 5> m_hash;
#endif
};

} // namespace WTF

using WTF::SHA1;
