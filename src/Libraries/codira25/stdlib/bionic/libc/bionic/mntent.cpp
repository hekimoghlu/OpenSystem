/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 14, 2023.
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
#include <mntent.h>
#include <string.h>

#include "bionic/pthread_internal.h"

mntent* getmntent(FILE* fp) {
  auto& tls = __get_bionic_tls();
  return getmntent_r(fp, &tls.mntent_buf, tls.mntent_strings, sizeof(tls.mntent_strings));
}

mntent* getmntent_r(FILE* fp, struct mntent* e, char* buf, int buf_len) {
  memset(e, 0, sizeof(*e));
  while (fgets(buf, buf_len, fp) != nullptr) {
    // Entries look like "proc /proc proc rw,nosuid,nodev,noexec,relatime 0 0".
    // That is: mnt_fsname mnt_dir mnt_type mnt_opts 0 0.
    int fsname0, fsname1, dir0, dir1, type0, type1, opts0, opts1;
    if (sscanf(buf, " %n%*s%n %n%*s%n %n%*s%n %n%*s%n %d %d",
               &fsname0, &fsname1, &dir0, &dir1, &type0, &type1, &opts0, &opts1,
               &e->mnt_freq, &e->mnt_passno) == 2) {
      e->mnt_fsname = &buf[fsname0];
      buf[fsname1] = '\0';

      e->mnt_dir = &buf[dir0];
      buf[dir1] = '\0';

      e->mnt_type = &buf[type0];
      buf[type1] = '\0';

      e->mnt_opts = &buf[opts0];
      buf[opts1] = '\0';

      return e;
    }
  }
  return nullptr;
}

FILE* setmntent(const char* path, const char* mode) {
  return fopen(path, mode);
}

int endmntent(FILE* fp) {
  if (fp != nullptr) {
    fclose(fp);
  }
  return 1;
}

char* hasmntopt(const struct mntent* mnt, const char* opt) {
  char* token = mnt->mnt_opts;
  char* const end = mnt->mnt_opts + strlen(mnt->mnt_opts);
  const size_t optLen = strlen(opt);

  while (token) {
    char* const tokenEnd = token + optLen;
    if (tokenEnd > end) break;

    if (memcmp(token, opt, optLen) == 0 &&
        (*tokenEnd == '\0' || *tokenEnd == ',' || *tokenEnd == '=')) {
      return token;
    }

    token = strchr(token, ',');
    if (token) token++;
  }

  return nullptr;
}
