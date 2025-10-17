/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 11, 2024.
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
enum BufferError: Error, Equatable, ConvertibleFromBufferError {
    case truncatedRead(UnsafeRawPointer?, UnsafeRawPointer?, Int)
    case stringError
    init(_ bufferReaderError: BufferError) {
        self = bufferReaderError
    }
}

protocol ConvertibleFromBufferError: Error {
    init(_ bufferReaderError: BufferError)
}

// Convenience methods for accessing mixed endian integers
internal enum Endian {
    case little
    case big
    // TODO: Make this correct for big endian platforms
    static var native: Endian {
        return .little
    }
    static var reverse: Endian {
        return .big
    }
}

extension FixedWidthInteger {
    func from(endian:Endian) -> Self {
        switch endian {
        case .little:
            return .init(littleEndian:self)
        case .big:
            return .init(bigEndian:self)
        }
    }
}

// These functions support reading from a range by enabling unaligned loads and endian swapping as
// necessary.

internal extension Slice where Base == UnsafeRawBufferPointer {
    mutating func read<T: FixedWidthInteger, E: ConvertibleFromBufferError>(endian: Endian = .little, as: T.Type = T.self, throwAs: E.Type = BufferError.self) throws(E) -> T {
        guard self.count >= MemoryLayout<T>.size else {
            let rebasedBuffer = UnsafeRawBufferPointer(rebasing:self)
            let startAddress = rebasedBuffer.baseAddress
            let endAddress: UnsafeRawPointer
            if let startAddress {
                endAddress = startAddress+rebasedBuffer.count
            } else {
                endAddress = UnsafeRawPointer(bitPattern:rebasedBuffer.count)!
            }
            throw E(.truncatedRead(startAddress, endAddress, MemoryLayout<T>.size))
        }
        let result = self.loadUnaligned(as:T.self).from(endian:endian)
        self = dropFirst(MemoryLayout<T>.size)
        return result
    }

    mutating func read<E: ConvertibleFromBufferError>(stringLength: Int, throwAs: E.Type = BufferError.self) throws(E) -> String {
        guard self.count >= stringLength else {
            let rebasedBuffer = UnsafeRawBufferPointer(rebasing:self)
            let startAddress = rebasedBuffer.baseAddress
            let endAddress: UnsafeRawPointer
            if let startAddress {
                endAddress = startAddress+rebasedBuffer.count
            } else {
                endAddress = UnsafeRawPointer(bitPattern:rebasedBuffer.count)!
            }
            throw E(.truncatedRead(startAddress, endAddress, stringLength))
        }
        let startIndex = indices.startIndex
        guard let result = String(bytes:base[startIndex..<startIndex+stringLength], encoding:.ascii) else { throw E(.stringError) }
        removeFirst(stringLength)
        return result
    }

    mutating func readUuid<E: ConvertibleFromBufferError>(throwAs: E.Type = BufferError.self) throws(E) -> UUID {
        guard self.count >= 16 else {
            let rebasedBuffer = UnsafeRawBufferPointer(rebasing:self)
            let startAddress = rebasedBuffer.baseAddress
            let endAddress: UnsafeRawPointer
            if let startAddress {
                endAddress = startAddress+rebasedBuffer.count
            } else {
                endAddress = UnsafeRawPointer(bitPattern:rebasedBuffer.count)!
            }
            throw E(.truncatedRead(startAddress, endAddress, 16))
        }
        let result =  self.withMemoryRebound(to: uuid_t.self) {
            return UUID(uuid:$0[0])
        }
        removeFirst(16)
        return result
    }
}

// Support for accessing unaligned arrays of integerss
struct UnalignedIntArray<T>: Collection where T: FixedWidthInteger {
    let bytes: Slice<UnsafeRawBufferPointer>
    let endian: Endian
    init(bytes: Slice<UnsafeRawBufferPointer>, endian:Endian = .little) {
        self.bytes = bytes
        self.endian = endian
    }

    // Sequence Protocol
    func makeIterator() -> Iterator {
        return Iterator(bytes, endian:endian)
    }

    // Collection protocol support
    var startIndex: Int { 0 }
    var endIndex: Int { bytes.count }
    subscript(index: Int) -> T {
        return  bytes.loadUnaligned(fromByteOffset:index*MemoryLayout<T>.size, as:T.self).from(endian:endian)
    }
    func index(after i: Int) -> Int {
        return i+1
    }

    struct Iterator: IteratorProtocol {
        var bytes: Slice<UnsafeRawBufferPointer>
        let endian: Endian
        init(_ bytes: Slice<UnsafeRawBufferPointer>, endian:Endian) {
            self.bytes = bytes
            self.endian = endian
        }

        mutating func next() -> T? {
            guard bytes.count >= MemoryLayout<T>.size else { return nil }
            let result = bytes.loadUnaligned(as:T.self).from(endian:endian)
            bytes = bytes.dropFirst(MemoryLayout<T>.size);
            return result
        }
    }
}
