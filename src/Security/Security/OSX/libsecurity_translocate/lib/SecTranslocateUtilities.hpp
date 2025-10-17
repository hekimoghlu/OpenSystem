/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 15, 2022.
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
/* Purpose:
 This header and its corresponding implementation are intended to house functionality that's useful
 throughtout SecTranslocate but isn't directly tied to the SPI or things that must be serialized.
 */

#ifndef SecTranslocateUtilities_hpp
#define SecTranslocateUtilities_hpp

#include <stdio.h>
#include <sys/param.h>
#include <sys/mount.h>
#include <security_utilities/unix++.h>

#include <string>
#include <vector>

#define NULLFS_FSTYPE "nullfs"

namespace Security {

using namespace Security::UnixPlusPlus;

namespace SecTranslocate {

using namespace std;

class ExtendedAutoFileDesc : public AutoFileDesc {
public:
    ExtendedAutoFileDesc():AutoFileDesc() {};
    ExtendedAutoFileDesc(const char *path, int flag = O_RDONLY, mode_t mode = 0666)
    : AutoFileDesc(path, flag, mode), mOriginalPath(path) { init(); }
    ExtendedAutoFileDesc(const std::string &path, int flag = O_RDONLY, mode_t mode = 0666)
    : AutoFileDesc(path, flag, mode), mOriginalPath(path) { init(); }
    ExtendedAutoFileDesc(int fd):AutoFileDesc(fd) { init(); }
    ExtendedAutoFileDesc(const ExtendedAutoFileDesc&) = default;
    ExtendedAutoFileDesc(ExtendedAutoFileDesc&&) = default;

    ExtendedAutoFileDesc & operator=(ExtendedAutoFileDesc&&);

    void open(const std::string &path, int flag = O_RDONLY, mode_t mode = 0666);
    
    bool isFileSystemType(const string &fsType) const;
    bool pathIsAbsolute() const;
    bool isMountPoint() const;
    bool isInPrefixDir(const string &prefixDir) const;
    string getFsType() const;
    string getMountPoint() const;
    string getMountFromPath() const;
    const string& getRealPath() const;
    fsid_t const getFsid() const;
    bool isQuarantined();
    bool isUserApproved();
    bool shouldTranslocate();
    bool isSandcastleProtected();
    
    // implicit destructor should call AutoFileDesc destructor. Nothing else to clean up.
private:
    void init();
    inline void notOpen() const { if(!isOpen()) UnixError::throwMe(EINVAL); };
    
    struct statfs mFsInfo;
    string mRealPath;
    string mOriginalPath;
    bool mQuarantineFetched;
    bool mQuarantined;
    uint32_t mQtn_flags;
    void fetchQuarantine();
};

//General utilities
string makeUUID();
void* checkedDlopen(const char* path, int mode);
void* checkedDlsym(void* handle, const char* symbol);

//Path parsing functions
vector<string> splitPath(const string &path);
string joinPath(vector<string>& path);
string joinPathUpTo(vector<string> &path, size_t index);

//File system utlities
string getRealPath(const string &path);
ExtendedAutoFileDesc getFDForDirectory(const string &directoryPath, bool *owned = NULL); //creates the directory if it can


//Translocation specific utilities
string translocationDirForUser();

} // namespace SecTranslocate
} // namespace Security


#endif /* SecTranslocateUtilities_hpp */
