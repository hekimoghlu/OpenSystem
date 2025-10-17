/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 17, 2023.
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
/*
 * This header is used to define error constants and names;
 * it might be included several times.
 */

#ifndef __BIONIC_ERRDEF
#error __BIONIC_ERRDEF not defined
#endif

__BIONIC_ERRDEF(0, "Success")
__BIONIC_ERRDEF(EPERM, "Operation not permitted")
__BIONIC_ERRDEF(ENOENT, "No such file or directory")
__BIONIC_ERRDEF(ESRCH, "No such process")
__BIONIC_ERRDEF(EINTR, "Interrupted system call")
__BIONIC_ERRDEF(EIO, "I/O error")
__BIONIC_ERRDEF(ENXIO, "No such device or address")
__BIONIC_ERRDEF(E2BIG, "Argument list too long")
__BIONIC_ERRDEF(ENOEXEC, "Exec format error")
__BIONIC_ERRDEF(EBADF, "Bad file descriptor")
__BIONIC_ERRDEF(ECHILD, "No child processes")
__BIONIC_ERRDEF(EAGAIN, "Try again")
__BIONIC_ERRDEF(ENOMEM, "Out of memory")
__BIONIC_ERRDEF(EACCES, "Permission denied")
__BIONIC_ERRDEF(EFAULT, "Bad address")
__BIONIC_ERRDEF(ENOTBLK, "Block device required")
__BIONIC_ERRDEF(EBUSY, "Device or resource busy")
__BIONIC_ERRDEF(EEXIST, "File exists")
__BIONIC_ERRDEF(EXDEV, "Cross-device link")
__BIONIC_ERRDEF(ENODEV, "No such device")
__BIONIC_ERRDEF(ENOTDIR, "Not a directory")
__BIONIC_ERRDEF(EISDIR, "Is a directory")
__BIONIC_ERRDEF(EINVAL, "Invalid argument")
__BIONIC_ERRDEF(ENFILE, "File table overflow")
__BIONIC_ERRDEF(EMFILE, "Too many open files")
__BIONIC_ERRDEF(ENOTTY, "Inappropriate ioctl for device")
__BIONIC_ERRDEF(ETXTBSY, "Text file busy")
__BIONIC_ERRDEF(EFBIG, "File too large")
__BIONIC_ERRDEF(ENOSPC, "No space left on device")
__BIONIC_ERRDEF(ESPIPE, "Illegal seek")
__BIONIC_ERRDEF(EROFS, "Read-only file system")
__BIONIC_ERRDEF(EMLINK, "Too many links")
__BIONIC_ERRDEF(EPIPE, "Broken pipe")
__BIONIC_ERRDEF(EDOM, "Math argument out of domain of func")
__BIONIC_ERRDEF(ERANGE, "Math result not representable")
__BIONIC_ERRDEF(EDEADLK, "Resource deadlock would occur")
__BIONIC_ERRDEF(ENAMETOOLONG, "File name too long")
__BIONIC_ERRDEF(ENOLCK, "No record locks available")
__BIONIC_ERRDEF(ENOSYS, "Function not implemented")
__BIONIC_ERRDEF(ENOTEMPTY, "Directory not empty")
__BIONIC_ERRDEF(ELOOP, "Too many symbolic links encountered")
__BIONIC_ERRDEF(ENOMSG, "No message of desired type")
__BIONIC_ERRDEF(EIDRM, "Identifier removed")
__BIONIC_ERRDEF(ECHRNG, "Channel number out of range")
__BIONIC_ERRDEF(EL2NSYNC, "Level 2 not synchronized")
__BIONIC_ERRDEF(EL3HLT, "Level 3 halted")
__BIONIC_ERRDEF(EL3RST, "Level 3 reset")
__BIONIC_ERRDEF(ELNRNG, "Link number out of range")
__BIONIC_ERRDEF(EUNATCH, "Protocol driver not attached")
__BIONIC_ERRDEF(ENOCSI, "No CSI structure available")
__BIONIC_ERRDEF(EL2HLT, "Level 2 halted")
__BIONIC_ERRDEF(EBADE, "Invalid exchange")
__BIONIC_ERRDEF(EBADR, "Invalid request descriptor")
__BIONIC_ERRDEF(EXFULL, "Exchange full")
__BIONIC_ERRDEF(ENOANO, "No anode")
__BIONIC_ERRDEF(EBADRQC, "Invalid request code")
__BIONIC_ERRDEF(EBADSLT, "Invalid slot")
__BIONIC_ERRDEF(EBFONT, "Bad font file format")
__BIONIC_ERRDEF(ENOSTR, "Device not a stream")
__BIONIC_ERRDEF(ENODATA, "No data available")
__BIONIC_ERRDEF(ETIME, "Timer expired")
__BIONIC_ERRDEF(ENOSR, "Out of streams resources")
__BIONIC_ERRDEF(ENONET, "Machine is not on the network")
__BIONIC_ERRDEF(ENOPKG, "Package not installed")
__BIONIC_ERRDEF(EREMOTE, "Object is remote")
__BIONIC_ERRDEF(ENOLINK, "Link has been severed")
__BIONIC_ERRDEF(EADV, "Advertise error")
__BIONIC_ERRDEF(ESRMNT, "Srmount error")
__BIONIC_ERRDEF(ECOMM, "Communication error on send")
__BIONIC_ERRDEF(EPROTO, "Protocol error")
__BIONIC_ERRDEF(EMULTIHOP, "Multihop attempted")
__BIONIC_ERRDEF(EDOTDOT, "RFS specific error")
__BIONIC_ERRDEF(EBADMSG, "Not a data message")
__BIONIC_ERRDEF(EOVERFLOW, "Value too large for defined data type")
__BIONIC_ERRDEF(ENOTUNIQ, "Name not unique on network")
__BIONIC_ERRDEF(EBADFD, "File descriptor in bad state")
__BIONIC_ERRDEF(EREMCHG, "Remote address changed")
__BIONIC_ERRDEF(ELIBACC, "Can not access a needed shared library")
__BIONIC_ERRDEF(ELIBBAD, "Accessing a corrupted shared library")
__BIONIC_ERRDEF(ELIBSCN, ".lib section in a.out corrupted")
__BIONIC_ERRDEF(ELIBMAX, "Attempting to link in too many shared libraries")
__BIONIC_ERRDEF(ELIBEXEC, "Cannot exec a shared library directly")
__BIONIC_ERRDEF(EILSEQ, "Illegal byte sequence")
__BIONIC_ERRDEF(ERESTART, "Interrupted system call should be restarted")
__BIONIC_ERRDEF(ESTRPIPE, "Streams pipe error")
__BIONIC_ERRDEF(EUSERS, "Too many users")
__BIONIC_ERRDEF(ENOTSOCK, "Socket operation on non-socket")
__BIONIC_ERRDEF(EDESTADDRREQ, "Destination address required")
__BIONIC_ERRDEF(EMSGSIZE, "Message too long")
__BIONIC_ERRDEF(EPROTOTYPE, "Protocol wrong type for socket")
__BIONIC_ERRDEF(ENOPROTOOPT, "Protocol not available")
__BIONIC_ERRDEF(EPROTONOSUPPORT, "Protocol not supported")
__BIONIC_ERRDEF(ESOCKTNOSUPPORT, "Socket type not supported")
__BIONIC_ERRDEF(EOPNOTSUPP, "Operation not supported on transport endpoint")
__BIONIC_ERRDEF(EPFNOSUPPORT, "Protocol family not supported")
__BIONIC_ERRDEF(EAFNOSUPPORT, "Address family not supported by protocol")
__BIONIC_ERRDEF(EADDRINUSE, "Address already in use")
__BIONIC_ERRDEF(EADDRNOTAVAIL, "Cannot assign requested address")
__BIONIC_ERRDEF(ENETDOWN, "Network is down")
__BIONIC_ERRDEF(ENETUNREACH, "Network is unreachable")
__BIONIC_ERRDEF(ENETRESET, "Network dropped connection because of reset")
__BIONIC_ERRDEF(ECONNABORTED, "Software caused connection abort")
__BIONIC_ERRDEF(ECONNRESET, "Connection reset by peer")
__BIONIC_ERRDEF(ENOBUFS, "No buffer space available")
__BIONIC_ERRDEF(EISCONN, "Transport endpoint is already connected")
__BIONIC_ERRDEF(ENOTCONN, "Transport endpoint is not connected")
__BIONIC_ERRDEF(ESHUTDOWN, "Cannot send after transport endpoint shutdown")
__BIONIC_ERRDEF(ETOOMANYREFS, "Too many references: cannot splice")
__BIONIC_ERRDEF(ETIMEDOUT, "Connection timed out")
__BIONIC_ERRDEF(ECONNREFUSED, "Connection refused")
__BIONIC_ERRDEF(EHOSTDOWN, "Host is down")
__BIONIC_ERRDEF(EHOSTUNREACH, "No route to host")
__BIONIC_ERRDEF(EALREADY, "Operation already in progress")
__BIONIC_ERRDEF(EINPROGRESS, "Operation now in progress")
__BIONIC_ERRDEF(ESTALE, "Stale NFS file handle")
__BIONIC_ERRDEF(EUCLEAN, "Structure needs cleaning")
__BIONIC_ERRDEF(ENOTNAM, "Not a XENIX named type file")
__BIONIC_ERRDEF(ENAVAIL, "No XENIX semaphores available")
__BIONIC_ERRDEF(EISNAM, "Is a named type file")
__BIONIC_ERRDEF(EREMOTEIO, "Remote I/O error")
__BIONIC_ERRDEF(EDQUOT, "Quota exceeded")
__BIONIC_ERRDEF(ENOMEDIUM, "No medium found")
__BIONIC_ERRDEF(EMEDIUMTYPE, "Wrong medium type")
__BIONIC_ERRDEF(ECANCELED, "Operation Canceled")
__BIONIC_ERRDEF(ENOKEY, "Required key not available")
__BIONIC_ERRDEF(EKEYEXPIRED, "Key has expired")
__BIONIC_ERRDEF(EKEYREVOKED, "Key has been revoked")
__BIONIC_ERRDEF(EKEYREJECTED, "Key was rejected by service")
__BIONIC_ERRDEF(EOWNERDEAD, "Owner died")
__BIONIC_ERRDEF(ENOTRECOVERABLE, "State not recoverable")
__BIONIC_ERRDEF(ERFKILL, "Operation not possible due to RF-kill")
__BIONIC_ERRDEF(EHWPOISON, "Memory page has hardware error")

#undef __BIONIC_ERRDEF
