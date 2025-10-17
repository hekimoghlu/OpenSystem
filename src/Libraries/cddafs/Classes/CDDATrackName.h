/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 9, 2021.
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
#ifndef __CDDA_TRACK_NAME_H__
#define __CDDA_TRACK_NAME_H__


//-----------------------------------------------------------------------------
//	Includes
//-----------------------------------------------------------------------------

#include "TBundle.h"


//-----------------------------------------------------------------------------
//	Class Declaration
//
//	CDDATrackName is the base class for all databases used. It provides
//	localized variants of the artist, title, and track names, as well as a
//	possible separator string used for diskarbitrationd
//-----------------------------------------------------------------------------

class CDDATrackName
{
	
	private:
		
		// Disable copy constructors
		CDDATrackName ( CDDATrackName &src );
		void operator = ( CDDATrackName &src );
		
		TBundle *		fBundle;
		
		CFStringRef		fTrackNameStringRef;
		CFStringRef		fAlbumStringRef;
		CFStringRef		fArtistStringRef;
		CFStringRef		fSeparatorStringRef;
		
	public:
		
		// Constructor
		CDDATrackName ( void );
		
		// Destructor
		virtual ~CDDATrackName ( void );		
		
		virtual SInt32			Init ( const char * bsdDevNode, const void * TOCData );
		
		virtual CFStringRef 	GetArtistName ( void );
		virtual CFStringRef 	GetAlbumName ( void );
		virtual CFStringRef 	GetSeparatorString ( void );
		virtual CFStringRef 	GetTrackName ( UInt8 trackNumber );
		
};


#endif	/* __CDDA_TRACK_NAME_H__ */
