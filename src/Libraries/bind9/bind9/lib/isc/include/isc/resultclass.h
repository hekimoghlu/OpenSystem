/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 16, 2023.
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
/* $Id: resultclass.h,v 1.20 2009/09/02 23:48:03 tbox Exp $ */

#ifndef ISC_RESULTCLASS_H
#define ISC_RESULTCLASS_H 1


/*! \file isc/resultclass.h
 * \brief Registry of Predefined Result Type Classes
 *
 * A result class number is an unsigned 16 bit number.  Each class may
 * contain up to 65536 results.  A result code is formed by adding the
 * result number within the class to the class number multiplied by 65536.
 *
 * Classes < 1024 are reserved for ISC use.
 * Result classes >= 1024 and <= 65535 are reserved for application use.
 */

#define ISC_RESULTCLASS_FROMNUM(num)		((num) << 16)
#define ISC_RESULTCLASS_TONUM(rclass)		((rclass) >> 16)
#define ISC_RESULTCLASS_SIZE			65536
#define ISC_RESULTCLASS_INCLASS(rclass, result) \
	((rclass) == ((result) & 0xFFFF0000))


#define	ISC_RESULTCLASS_ISC		ISC_RESULTCLASS_FROMNUM(0)
#define	ISC_RESULTCLASS_DNS		ISC_RESULTCLASS_FROMNUM(1)
#define	ISC_RESULTCLASS_DST		ISC_RESULTCLASS_FROMNUM(2)
#define	ISC_RESULTCLASS_DNSRCODE	ISC_RESULTCLASS_FROMNUM(3)
#define	ISC_RESULTCLASS_OMAPI		ISC_RESULTCLASS_FROMNUM(4)
#define	ISC_RESULTCLASS_ISCCC		ISC_RESULTCLASS_FROMNUM(5)
#define	ISC_RESULTCLASS_DHCP		ISC_RESULTCLASS_FROMNUM(6)
#define	ISC_RESULTCLASS_PK11		ISC_RESULTCLASS_FROMNUM(7)

#endif /* ISC_RESULTCLASS_H */
