/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 5, 2023.
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
#ifndef FileManager_h
#define FileManager_h

#if !TARGET_OS_EXCLAVEKIT
#include "UUID.h"
#include "Defines.h"
#include "Allocator.h"
#include "OrderedMap.h"
#include "DyldDelegates.h"
#if !BUILDING_DYLD
#include <os/lock.h>
#endif

namespace dyld4 {

using lsl::UUID;
using lsl::UniquePtr;
using lsl::Allocator;
using lsl::OrderedMap;

struct FileManager;

struct VIS_HIDDEN FileRecord {
    FileRecord()                                = default;
    FileRecord(const FileRecord&other);
    FileRecord(FileRecord&& other);
    ~FileRecord();
    FileRecord& operator=(const FileRecord& other);
    FileRecord& operator=(FileRecord&& other);

    uint64_t        objectID() const;
    uint64_t        mtime() const;
    size_t          size() const;
    const UUID&     volume() const;

    int             open(int flags);
    void            close();

    bool            exists() const;
    const char*     getPath() const;
    bool            persistent() const;
    FileManager&    fileManager() const;

    friend void swap(FileRecord& x, FileRecord& y) {
        x.swap(y);
    }
private:
    friend FileManager;
    FileRecord(FileManager& fileManager, uint64_t objectID, uint64_t device, uint64_t mtime);
    FileRecord(FileManager& fileManager, const UUID& VID, uint64_t objectID);
    FileRecord(FileManager& fileManager, const struct stat& sb);
    FileRecord(FileManager& fileManager, UniquePtr<const char>&& filePath);
    void swap(FileRecord& other);
    void stat() const;

    FileManager*                        _fileManager    = nullptr;
    mutable uint64_t                    _objectID       = 0;
    mutable uint64_t                    _device         = 0;
    mutable UUID                        _volume;
    mutable lsl::UniquePtr<const char>  _path;
    mutable size_t                      _size           = 0;
    mutable uint64_t                    _mtime          = 0;
    int                                 _fd             = -1;
    mutable int                         _statResult     = 1; // Tri-state: 1 not stated, 0 successful stat, -1 failed stat
    mutable mode_t                      _mode           = 0;
    mutable bool                        _valid          = true;
};

struct VIS_HIDDEN FileManager {
                    FileManager()                       = delete;
                    FileManager(const FileManager&)     = delete;
                    FileManager(FileManager&&)          = delete;
    FileManager&    operator=(const FileManager& O)     = delete;
    FileManager&    operator=(FileManager&& O)          = delete;
    FileManager(Allocator& allocator, const SyscallDelegate* syscall);
    FileManager(Allocator& allocator);

    FileRecord      fileRecordForPath(Allocator& allocator, const char* filePath);
    FileRecord      fileRecordForStat(const struct stat& sb);
    FileRecord      fileRecordForVolumeUUIDAndObjID(const UUID& VID, uint64_t objectID);
    FileRecord      fileRecordForVolumeDevIDAndObjID(uint64_t device, uint64_t objectID);
    FileRecord      fileRecordForFileID(const FileID& fileID);
    const UUID     uuidForFileSystem(uint64_t fsid) const;
    uint64_t        fsidForUUID(const UUID& uuid) const;
    friend void     swap(FileManager& x, FileManager& y) {
        x.swap(y);
    }
private:
    friend FileRecord;

    void    swap(FileManager& other);
    ssize_t fsgetpath(char result[], size_t resultBufferSize, uint64_t fsID, uint64_t objID) const;
    int     getfsstat(struct statfs *buf, int bufsize, int flags) const;

    int getattrlist(const char* path, struct attrlist * attrList, void * attrBuf, size_t attrBufSize, uint32_t options) const;

    void reloadFSInfos() const;
    UniquePtr<char> getPath(const lsl::UUID& VID, uint64_t OID);
    UniquePtr<char> getPath(uint64_t fsid, uint64_t OID);

    const SyscallDelegate*                          _syscall        = nullptr;
    Allocator*                                      _allocator      = nullptr;
    mutable UniquePtr<OrderedMap<uint64_t,UUID>>    _fsUUIDMap      = nullptr;
    //FIXME: We should probably have a more generic lock abstraction for locks we only need when not building dyld
    template<typename F>
    auto withFSInfoLock(F work) const
    {
        #if BUILDING_DYLD
        return work();
        #else
        os_unfair_lock_lock(&_fsUUIDMapLock);
        auto result = work();
        os_unfair_lock_unlock(&_fsUUIDMapLock);
        return result;
        #endif
    }
#if !BUILDING_DYLD
    mutable os_unfair_lock_s                        _fsUUIDMapLock  = OS_UNFAIR_LOCK_INIT;
#endif
};

}; /* namespace dyld4 */

#endif //!TARGET_OS_EXCLAVEKIT
#endif /* FileManager_h */
