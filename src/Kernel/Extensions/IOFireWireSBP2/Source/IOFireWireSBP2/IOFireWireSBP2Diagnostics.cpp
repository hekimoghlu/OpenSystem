/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 14, 2025.
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
#include "IOFireWireSBP2Diagnostics.h"

OSDefineMetaClassAndStructors(IOFireWireSBP2Diagnostics, OSObject)

OSObject * IOFireWireSBP2Diagnostics::createDiagnostics( void )
{
	IOFireWireSBP2Diagnostics *	diagnostics;
	
	diagnostics = new IOFireWireSBP2Diagnostics;
	if( diagnostics && !diagnostics->init() )
	{
		diagnostics->release();
		diagnostics = NULL;
	}
	
	return diagnostics;
}

bool IOFireWireSBP2Diagnostics::serialize( OSSerialize * s ) const
{
	OSDictionary *	dictionary;
	bool			ok;
	
	dictionary = OSDictionary::withCapacity( 4 );
	if( !dictionary )
		return false;
		
	/////////
	
	updateNumberEntry( dictionary, fExecutedORBCount, "Executed ORB Count" );
	
	/////////
	
	ok = dictionary->serialize(s);
	dictionary->release();
	
	return ok;
}
	
void IOFireWireSBP2Diagnostics::updateNumberEntry( OSDictionary * dictionary, UInt32 value, const char * name )
{
	OSNumber *	number;
	
	number = OSNumber::withNumber( value, 32 );
	if( !number )
		return;
		
	dictionary->setObject( name, number );
	number->release();
}

///////////////////////////

void IOFireWireSBP2Diagnostics::incrementExecutedORBCount( void )
{
	fExecutedORBCount++;
}


