/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 17, 2025.
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
// implementationOnly so we don't leak the c types into the public interface
@_implementationOnly import dyld_cache_module
//@_implementationOnly import MachO_Private.dyld_cache_format

enum SharedCacheRuntimeError: Error {
    /// ran out of buffer while parsing
    case outOfBuffer(limit: Int, required: Int)
}

// Swift API to be used exclusively by the ExclaveKit loader
public struct MappingInfo {
    public let address     : UInt64
    public let size        : UInt64
    public let fileOffset  : UInt64
    public let slideOffset : UInt64
    public let slideSize   : UInt64
    public let flags       : UInt64
    public let maxProt     : UInt32
    public let initProt    : UInt32
}

public struct CodeSignatureInfo {
    public let offset : UInt64
    public let size   : UInt64
}

private func loadFromRawBuffer<T>(cacheBuffer: UnsafeRawBufferPointer, offset: Int, end: Int) throws -> T {
    guard cacheBuffer.count > end else {
        throw SharedCacheRuntimeError.outOfBuffer(
            limit: cacheBuffer.count,
            required: end
        )
    }
    return cacheBuffer.load(fromByteOffset: offset,
                                  as: T.self)
}
public func hasValidMagic(cacheBuffer: UnsafeRawBufferPointer) throws -> Bool {
    let headerSize = MemoryLayout<dyld_cache_header>.stride
    let header : dyld_cache_header = try loadFromRawBuffer(cacheBuffer: cacheBuffer,
                                                       offset: 0,
                                                       end: headerSize)
    let isValidMagic = withUnsafeBytes(of: header.magic) {
        buffer in
        let magicString = String(decoding: [UInt8](buffer), as: UTF8.self)
        return magicString == "dyld_v1  arm64e\0"
    }
    return isValidMagic
}

public func getPlatform(cacheBuffer: UnsafeRawBufferPointer) throws -> UInt32 {
    let headerSize = MemoryLayout<dyld_cache_header>.stride
    let header : dyld_cache_header = try loadFromRawBuffer(cacheBuffer: cacheBuffer,
                                                       offset: 0,
                                                       end: headerSize)
    return header.platform
 }

public func getUUID(cacheBuffer: UnsafeRawBufferPointer) throws -> [UInt8] {
    let headerSize = MemoryLayout<dyld_cache_header>.stride
    let header : dyld_cache_header = try loadFromRawBuffer(cacheBuffer: cacheBuffer,
                                                       offset: 0,
                                                       end: headerSize)
    return withUnsafeBytes(of: header.uuid) { buf in
            [UInt8](buf)
    }
 }


public func getMappingsInfo(cacheBuffer: UnsafeRawBufferPointer) throws -> [MappingInfo] {
    let headerSize = MemoryLayout<dyld_cache_header>.stride
    let header : dyld_cache_header = try loadFromRawBuffer(cacheBuffer: cacheBuffer,
                                                       offset: 0,
                                                       end: headerSize)

    let mappings = try (0..<Int(header.mappingWithSlideCount)).map { index in
        let structSize = MemoryLayout<dyld_cache_mapping_and_slide_info>.stride
        let offset = Int(header.mappingWithSlideOffset) + index * structSize
        let mappingEnd = offset + structSize
        let m : dyld_cache_mapping_and_slide_info = try loadFromRawBuffer(cacheBuffer: cacheBuffer,
                                                                          offset: offset,
                                                                          end: mappingEnd)
        return MappingInfo(address: m.address, size: m.size, fileOffset: m.fileOffset,
                           slideOffset: m.slideInfoFileOffset, slideSize: m.slideInfoFileSize,
                           flags: m.flags, maxProt: m.maxProt, initProt: m.initProt)
    }
    return mappings
}

public func getCodeSignatureInfo(cacheBuffer: UnsafeRawBufferPointer)throws -> CodeSignatureInfo {
    let headerSize = MemoryLayout<dyld_cache_header>.stride
    let header : dyld_cache_header = try loadFromRawBuffer(cacheBuffer: cacheBuffer,
                                                       offset: 0,
                                                       end: headerSize)
    return CodeSignatureInfo(offset: header.codeSignatureOffset, size: header.codeSignatureSize)
}
