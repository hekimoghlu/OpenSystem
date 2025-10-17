/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 23, 2024.
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
#ifndef _IMAGEBOOT_H_
#define _IMAGEBOOT_H_

struct vnode;

typedef enum imageboot_type {
	IMAGEBOOT_NONE,
	IMAGEBOOT_DMG,
	IMAGEBOOT_LOCKER,
} imageboot_type_t;

imageboot_type_t        imageboot_needed(void);
bool    imageboot_desired(void);
void    imageboot_setup(imageboot_type_t type);
int     imageboot_format_is_valid(const char *root_path);
int     imageboot_mount_image(const char *root_path, int height, imageboot_type_t type);
int     imageboot_pivot_image(const char *image_path, imageboot_type_t type, const char *mount_path,
    const char *outgoing_root_path, const bool rooted_dmg, const bool skip_signature_check);
int     imageboot_read_file_pageable(const char *path, void **bufp, size_t *bufszp); /* use kmem_free(kernel_map, ...) */
int     imageboot_read_file(const char *path, void **bufp, size_t *bufszp, off_t *fsizep);
int     imageboot_read_file_from_offset(const char *path, off_t offset, void **bufp, size_t *bufszp);

struct vnode *
imgboot_get_image_file(const char *path, off_t *fsize, int *errp);

#define IMAGEBOOT_CONTAINER_ARG         "container-dmg"
#define IMAGEBOOT_ROOT_ARG              "root-dmg"
#define IMAGEBOOT_AUTHROOT_ARG          "auth-root-dmg"

//IMAGEBOOT images are capped at 2.5GB
#define IMAGEBOOT_MAX_FILESIZE          (2684354560ULL)
//limit certain kalloc calls to 2GB
#define IMAGEBOOT_MAX_KALLOCSIZE        (2147483648ULL)

#endif
