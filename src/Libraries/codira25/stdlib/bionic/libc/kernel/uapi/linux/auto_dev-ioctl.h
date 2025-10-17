/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 7, 2024.
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
#ifndef _UAPI_LINUX_AUTO_DEV_IOCTL_H
#define _UAPI_LINUX_AUTO_DEV_IOCTL_H
#include <linux/auto_fs.h>
#include <linux/string.h>
#define AUTOFS_DEVICE_NAME "autofs"
#define AUTOFS_DEV_IOCTL_VERSION_MAJOR 1
#define AUTOFS_DEV_IOCTL_VERSION_MINOR 1
#define AUTOFS_DEV_IOCTL_SIZE sizeof(struct autofs_dev_ioctl)
struct args_protover {
  __u32 version;
};
struct args_protosubver {
  __u32 sub_version;
};
struct args_openmount {
  __u32 devid;
};
struct args_ready {
  __u32 token;
};
struct args_fail {
  __u32 token;
  __s32 status;
};
struct args_setpipefd {
  __s32 pipefd;
};
struct args_timeout {
  __u64 timeout;
};
struct args_requester {
  __u32 uid;
  __u32 gid;
};
struct args_expire {
  __u32 how;
};
struct args_askumount {
  __u32 may_umount;
};
struct args_ismountpoint {
  union {
    struct args_in {
      __u32 type;
    } in;
    struct args_out {
      __u32 devid;
      __u32 magic;
    } out;
  };
};
struct autofs_dev_ioctl {
  __u32 ver_major;
  __u32 ver_minor;
  __u32 size;
  __s32 ioctlfd;
  union {
    struct args_protover protover;
    struct args_protosubver protosubver;
    struct args_openmount openmount;
    struct args_ready ready;
    struct args_fail fail;
    struct args_setpipefd setpipefd;
    struct args_timeout timeout;
    struct args_requester requester;
    struct args_expire expire;
    struct args_askumount askumount;
    struct args_ismountpoint ismountpoint;
  };
  char path[];
};
enum {
  AUTOFS_DEV_IOCTL_VERSION_CMD = 0x71,
  AUTOFS_DEV_IOCTL_PROTOVER_CMD,
  AUTOFS_DEV_IOCTL_PROTOSUBVER_CMD,
  AUTOFS_DEV_IOCTL_OPENMOUNT_CMD,
  AUTOFS_DEV_IOCTL_CLOSEMOUNT_CMD,
  AUTOFS_DEV_IOCTL_READY_CMD,
  AUTOFS_DEV_IOCTL_FAIL_CMD,
  AUTOFS_DEV_IOCTL_SETPIPEFD_CMD,
  AUTOFS_DEV_IOCTL_CATATONIC_CMD,
  AUTOFS_DEV_IOCTL_TIMEOUT_CMD,
  AUTOFS_DEV_IOCTL_REQUESTER_CMD,
  AUTOFS_DEV_IOCTL_EXPIRE_CMD,
  AUTOFS_DEV_IOCTL_ASKUMOUNT_CMD,
  AUTOFS_DEV_IOCTL_ISMOUNTPOINT_CMD,
};
#define AUTOFS_DEV_IOCTL_VERSION _IOWR(AUTOFS_IOCTL, AUTOFS_DEV_IOCTL_VERSION_CMD, struct autofs_dev_ioctl)
#define AUTOFS_DEV_IOCTL_PROTOVER _IOWR(AUTOFS_IOCTL, AUTOFS_DEV_IOCTL_PROTOVER_CMD, struct autofs_dev_ioctl)
#define AUTOFS_DEV_IOCTL_PROTOSUBVER _IOWR(AUTOFS_IOCTL, AUTOFS_DEV_IOCTL_PROTOSUBVER_CMD, struct autofs_dev_ioctl)
#define AUTOFS_DEV_IOCTL_OPENMOUNT _IOWR(AUTOFS_IOCTL, AUTOFS_DEV_IOCTL_OPENMOUNT_CMD, struct autofs_dev_ioctl)
#define AUTOFS_DEV_IOCTL_CLOSEMOUNT _IOWR(AUTOFS_IOCTL, AUTOFS_DEV_IOCTL_CLOSEMOUNT_CMD, struct autofs_dev_ioctl)
#define AUTOFS_DEV_IOCTL_READY _IOWR(AUTOFS_IOCTL, AUTOFS_DEV_IOCTL_READY_CMD, struct autofs_dev_ioctl)
#define AUTOFS_DEV_IOCTL_FAIL _IOWR(AUTOFS_IOCTL, AUTOFS_DEV_IOCTL_FAIL_CMD, struct autofs_dev_ioctl)
#define AUTOFS_DEV_IOCTL_SETPIPEFD _IOWR(AUTOFS_IOCTL, AUTOFS_DEV_IOCTL_SETPIPEFD_CMD, struct autofs_dev_ioctl)
#define AUTOFS_DEV_IOCTL_CATATONIC _IOWR(AUTOFS_IOCTL, AUTOFS_DEV_IOCTL_CATATONIC_CMD, struct autofs_dev_ioctl)
#define AUTOFS_DEV_IOCTL_TIMEOUT _IOWR(AUTOFS_IOCTL, AUTOFS_DEV_IOCTL_TIMEOUT_CMD, struct autofs_dev_ioctl)
#define AUTOFS_DEV_IOCTL_REQUESTER _IOWR(AUTOFS_IOCTL, AUTOFS_DEV_IOCTL_REQUESTER_CMD, struct autofs_dev_ioctl)
#define AUTOFS_DEV_IOCTL_EXPIRE _IOWR(AUTOFS_IOCTL, AUTOFS_DEV_IOCTL_EXPIRE_CMD, struct autofs_dev_ioctl)
#define AUTOFS_DEV_IOCTL_ASKUMOUNT _IOWR(AUTOFS_IOCTL, AUTOFS_DEV_IOCTL_ASKUMOUNT_CMD, struct autofs_dev_ioctl)
#define AUTOFS_DEV_IOCTL_ISMOUNTPOINT _IOWR(AUTOFS_IOCTL, AUTOFS_DEV_IOCTL_ISMOUNTPOINT_CMD, struct autofs_dev_ioctl)
#endif
