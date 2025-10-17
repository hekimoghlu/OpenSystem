/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 25, 2022.
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
#ifndef	_IO_ATA_FAMILY_PRIV_H
#define	_IO_ATA_FAMILY_PRIV_H


/*!
 
 @header IOATAFamilyPriv.h
 @discussion contains various private definitions and constants for use in the IOATAFamily and clients. Header Doc is incomplete at this point, but file is heavily commented.
 
 */


/* The ATA private Event codes */
/* sent when calling the device driver's event handler */

enum ataPrivateEventCode
{
	
	kATANewMediaEvent			= 0x80		/* Someone has added new media to the drive */
	
};


#endif	//_IO_ATA_FAMILY_PRIV_H

