/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 4, 2023.
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
#ifdef MULTIUSER

/* three known bits: */
#define ACL_EXEC 0		
#define ACL_WRITE 1
#define ACL_READ 2

#define ACL_BITS_PER_CMD 1	/* for comm.h */
#define ACL_BITS_PER_WIN 3	/* for window.h */

#define USER_CHUNK 8

#define ACLBYTE(data, w)   ((data)[(w) >> 3])
#define ACLBIT(w)   (0x80 >> ((w) & 7))

typedef unsigned char * AclBits;

/*
 * How a user joins a group.
 * Here is the node to construct one list per user.
 */
struct aclusergroup
{
  struct acluser *u;	/* the user who borrows us his rights */
  struct aclusergroup *next;
};
#endif /* MULTIUSER */

/***************
 *  ==> user.h
 */

/*
 * a copy buffer
 */
struct plop
{
  char *buf;
  int len;
#ifdef ENCODINGS
  int enc;
#endif
};

/*
 * A User has a list of groups, and points to other users.  
 * users is the User entry of the session owner (creator)
 * and anchors all other users. Add/Delete users there.
 */
typedef struct acluser
{
  struct acluser *u_next;		/* continue the main user list */
  char u_name[20+1];		/* login name how he showed up */
  char *u_password;		/* his password (may be NullStr). */
  int  u_checkpassword;		/* nonzero if this u_password is valid */
  int  u_detachwin;		/* the window where he last detached */
  int  u_detachotherwin;	/* window that was "other" when he detached */
  int  u_Esc, u_MetaEsc;	/* the users screen escape character */
#ifdef COPY_PASTE
  struct plop u_plop;
#endif
#ifdef MULTIUSER
  int u_id;			/* a uniq index in the bitfields. */
  AclBits u_umask_w_bits[ACL_BITS_PER_WIN];	/* his window create umask */
  struct aclusergroup *u_group;	/* linked list of pointers to other users */
#endif
} User;

extern int DefaultEsc, DefaultMetaEsc;

