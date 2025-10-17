/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 15, 2023.
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
import os
import System
@_implementationOnly import Dyld_Internal

@available(macOS 13.0, *)
struct AtlasCache {
    var uuidMap = [UUID : BPList.UnsafeRawDictionary]()
    var pathMap = [String : BPList.UnsafeRawDictionary]()
    var buffers = [Data]()
    let lock = OSAllocatedUnfairLock()
    mutating func getCachePlist(uuid: UUID?, path: FilePath?, forceScavenge: Bool) -> BPList.UnsafeRawDictionary? {
        return try? lock.withLockUnchecked { () -> BPList.UnsafeRawDictionary? in
            //FIXME: Delegate support
            if let uuid, let result = uuidMap[uuid] {
                return result
            }
            guard let path else {
                return nil
            }
            if !forceScavenge, let result = pathMap[path.string] {
                return result
            }
            guard let atlasFileName = path.lastComponent?.stem.appending(".atlas") else {
                return  nil
            }
            var embeddedData = try? Data(contentsOf:URL(fileURLWithPath:path.removingLastComponent().appending(atlasFileName).string), options:.mappedIfSafe)
            var scavengedData: Data? = nil
            if forceScavenge || embeddedData == nil {
                embeddedData = nil
                var scavengedBufferSize = UInt64(0)
                if let scavegedBuffer =  scavengeCache(path.string, &scavengedBufferSize) {
                    scavengedData = Data(bytesNoCopy:scavegedBuffer, count:Int(scavengedBufferSize), deallocator:.free)
                }
            }
            let data = embeddedData ?? scavengedData
            guard let data else { return nil }
            buffers.append(data)
            var archive  = try AARDecoder(data:data)
            let archivePath = uuid?.cacheAtlasArchivePath ?? "caches/names/\(path.lastComponent!.string).plist"
            guard let plistData = try? archive.data(path: archivePath),
                  let atlasesDict = try? BPList(data:plistData).asDictionary(),
                  let byNameDict = try atlasesDict[.string("names")]?.asDictionary(),
                  let byUuidDict = try atlasesDict[.string("uuids")]?.asDictionary() else {
                return nil
            }
            var skipValidation = false
            if scavengedData == nil {
                for prefix in SharedCache.sharedCachePaths {
                    if path.string.hasPrefix(prefix) {
                        // The shared cache is coming from snapshot protected storage, skip validation
                        skipValidation = true
                        break
                    }
                }
            }
            if !skipValidation {
                do {
                    for (_,atlas) in byNameDict {
                        guard let atlasDict = try? atlas.asDictionary() else {
                            throw AtlasError.placeHolder
                        }
                        try SharedCache.Impl.validate(bplist:atlasDict)
                    }
                    for (_,atlas) in byUuidDict {
                        guard let atlasDict = try? atlas.asDictionary() else {
                            throw AtlasError.placeHolder
                        }
                        try SharedCache.Impl.validate(bplist:atlasDict)
                    }
                } catch {
                    return nil
                }
            }
            for (key,atlas) in byUuidDict {
                guard let atlasUUID = UUID(uuidString: key.asString()) else {
                    continue
                }
                if uuidMap[atlasUUID] == nil {
                    uuidMap[atlasUUID] = try! atlas.asDictionary()
                }
            }
            let dirname = path.removingLastComponent()
            for (atlasName,atlas) in byNameDict {
                let fullPath = dirname.appending(atlasName.asString()).string
                let atlasDict = try! atlas.asDictionary()
                if pathMap[fullPath] == nil {
                    pathMap[fullPath] = atlasDict
                }
            }
            if let uuid {
                return uuidMap[uuid]
            }
            return pathMap[path.string]
        }
    }
}

@available(macOS 13.0, *)
fileprivate var bufferCache = AtlasCache()

internal extension Snapshot {
    static func findCacheBPlist(uuid: UUID?, path: FilePath?, forceScavenge: Bool = false) -> BPList.UnsafeRawDictionary? {
        if #available(macOS 13.0, *) {
            guard let plist = bufferCache.getCachePlist(uuid:uuid, path:path, forceScavenge:forceScavenge) else {
                return nil
            }
            return plist
        } else {
            fatalError("This is not supported for backdeployment, but needs to build with a reduced minOS in some environments")
        }
    }
}
