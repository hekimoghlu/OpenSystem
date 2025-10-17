/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 15, 2021.
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

/**
 * @file sys/xattr.h
 * @brief Extended attribute functions.
 */

#include <sys/cdefs.h>

#include <linux/xattr.h>
#include <sys/types.h>

__BEGIN_DECLS

/**
 * [fsetxattr(2)](https://man7.org/linux/man-pages/man2/fsetxattr.2.html)
 * sets an extended attribute on the file referred to by the given file
 * descriptor.
 *
 * A `size` of 0 can be used to set an empty value, in which case `value` is
 * ignored and may be null. Setting an xattr to an empty value is not the same
 * as removing an xattr; see removexattr() for the latter operation.
 *
 * Valid flags are `XATTR_CREATE` and `XATTR_REPLACE`.
 *
 * Returns 0 on success and returns -1 and sets `errno` on failure.
 */
int fsetxattr(int __fd, const char* _Nonnull __name, const void* _Nullable __value, size_t __size, int __flags);

/**
 * [setxattr(2)](https://man7.org/linux/man-pages/man2/setxattr.2.html)
 * sets an extended attribute on the file referred to by the given path.
 *
 * A `size` of 0 can be used to set an empty value, in which case `value` is
 * ignored and may be null. Setting an xattr to an empty value is not the same
 * as removing an xattr; see removexattr() for the latter operation.
 *
 * Valid flags are `XATTR_CREATE` and `XATTR_REPLACE`.
 *
 * Returns 0 on success and returns -1 and sets `errno` on failure.
 */
int setxattr(const char* _Nonnull __path, const char* _Nonnull __name, const void* _Nullable __value, size_t __size, int __flags);

/**
 * [lsetxattr(2)](https://man7.org/linux/man-pages/man2/lsetxattr.2.html)
 * sets an extended attribute on the file referred to by the given path, which
 * is the link itself rather than its target in the case of a symbolic link.
 *
 * A `size` of 0 can be used to set an empty value, in which case `value` is
 * ignored and may be null. Setting an xattr to an empty value is not the same
 * as removing an xattr; see removexattr() for the latter operation.
 *
 * Valid flags are `XATTR_CREATE` and `XATTR_REPLACE`.
 *
 * Returns 0 on success and returns -1 and sets `errno` on failure.
 */
int lsetxattr(const char* _Nonnull __path, const char* _Nonnull __name, const void* _Nullable __value, size_t __size, int __flags);

/**
 * [fgetxattr(2)](https://man7.org/linux/man-pages/man2/fgetxattr.2.html)
 * gets an extended attribute on the file referred to by the given file
 * descriptor.
 *
 * A `size` of 0 can be used to query the current length, in which case `value` is ignored and may be null.
 *
 * Returns the non-negative length of the value on success, or
 * returns -1 and sets `errno` on failure.
 */
ssize_t fgetxattr(int __fd, const char* _Nonnull __name, void* _Nullable __value, size_t __size);

/**
 * [getxattr(2)](https://man7.org/linux/man-pages/man2/getxattr.2.html)
 * gets an extended attribute on the file referred to by the given path.
 *
 * A `size` of 0 can be used to query the current length, in which case `value` is ignored and may be null.
 *
 * Returns the non-negative length of the value on success, or
 * returns -1 and sets `errno` on failure.
 */
ssize_t getxattr(const char* _Nonnull __path, const char* _Nonnull __name, void* _Nullable __value, size_t __size);

/**
 * [lgetxattr(2)](https://man7.org/linux/man-pages/man2/lgetxattr.2.html)
 * gets an extended attribute on the file referred to by the given path, which
 * is the link itself rather than its target in the case of a symbolic link.
 *
 * A `size` of 0 can be used to query the current length, in which case `value` is ignored and may be null.
 *
 * Returns the non-negative length of the value on success, or
 * returns -1 and sets `errno` on failure.
 */
ssize_t lgetxattr(const char* _Nonnull __path, const char* _Nonnull __name, void* _Nullable __value, size_t __size);

/**
 * [flistxattr(2)](https://man7.org/linux/man-pages/man2/flistxattr.2.html)
 * lists the extended attributes on the file referred to by the given file
 * descriptor.
 *
 * A `size` of 0 can be used to query the current length, in which case `list` is ignored and may be null.
 *
 * Returns the non-negative length of the list on success, or
 * returns -1 and sets `errno` on failure.
 */
ssize_t flistxattr(int __fd, char* _Nullable __list, size_t __size);

/**
 * [listxattr(2)](https://man7.org/linux/man-pages/man2/listxattr.2.html)
 * lists the extended attributes on the file referred to by the given path.
 *
 * A `size` of 0 can be used to query the current length, in which case `list` is ignored and may be null.
 *
 * Returns the non-negative length of the list on success, or
 * returns -1 and sets `errno` on failure.
 */
ssize_t listxattr(const char* _Nonnull __path, char* _Nullable __list, size_t __size);

/**
 * [llistxattr(2)](https://man7.org/linux/man-pages/man2/llistxattr.2.html)
 * lists the extended attributes on the file referred to by the given path, which
 * is the link itself rather than its target in the case of a symbolic link.
 *
 * A `size` of 0 can be used to query the current length, in which case `list` is ignored and may be null.
 *
 * Returns the non-negative length of the list on success, or
 * returns -1 and sets `errno` on failure.
 */
ssize_t llistxattr(const char* _Nonnull __path, char* _Nullable __list, size_t __size);

/**
 * [fremovexattr(2)](https://man7.org/linux/man-pages/man2/fremovexattr.2.html)
 * removes an extended attribute on the file referred to by the given file
 * descriptor.
 *
 * Returns 0 on success and returns -1 and sets `errno` on failure.
 */
int fremovexattr(int __fd, const char* _Nonnull __name);

/**
 * [lremovexattr(2)](https://man7.org/linux/man-pages/man2/lremovexattr.2.html)
 * removes an extended attribute on the file referred to by the given path, which
 * is the link itself rather than its target in the case of a symbolic link.
 *
 * Returns 0 on success and returns -1 and sets `errno` on failure.
 */
int lremovexattr(const char* _Nonnull __path, const char* _Nonnull __name);

/**
 * [removexattr(2)](https://man7.org/linux/man-pages/man2/removexattr.2.html)
 * removes an extended attribute on the file referred to by the given path.
 *
 * Returns 0 on success and returns -1 and sets `errno` on failure.
 */
int removexattr(const char* _Nonnull __path, const char* _Nonnull __name);

__END_DECLS
