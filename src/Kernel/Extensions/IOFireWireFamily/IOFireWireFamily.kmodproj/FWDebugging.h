/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 15, 2025.
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
#import "FWTracepoints.h"
 
// the controls 

// *** START HERE: disable debug logging
#define FWLOGGING 0
#define FWASSERTS 1

///////////////////////////////////////////

#if FWLOGGING
	#define FWKLOG( x )					IOLog x
	#define DebugLog( x... )			IOLog( x ) ;
	#define DebugLogCond( x, y... ) 	{ if (x) DebugLog( y ) ; }
#else
	#define FWKLOG( x... )				do {} while (0)
	#define DebugLog( x... )			do {} while (0)
	#define DebugLogCond( x, y... )		do {} while (0)
#endif

#if FWLOCALLOGGING
#define FWLOCALKLOG(x) IOLog x
#else
#define FWLOCALKLOG(x) do {} while (0)
#endif

#if FWASSERTS
#define FWKLOGASSERT(a) { if(!(a)) { IOLog( "File %s, line %d: assertion '%s' failed.\n", __FILE__, __LINE__, #a); } }
#else
#define FWKLOGASSERT(a) do {} while (0)
#endif

#if FWASSERTS
#define FWPANICASSERT(a) { if(!(a)) { panic( "File "__FILE__", line %d: assertion '%s' failed.\n", __LINE__, #a); } }
#else
#define FWPANICASSERT(a) do {} while (0)
#endif

#if FWASSERTS
#define FWASSERTINGATE(a) { if(!((a)->inGate())) { IOLog( "File "__FILE__", line %d: warning - workloop lock is not held.\n", __LINE__); } }
#else
#define FWASSERTINGATE(a) do {} while (0)
#endif

#define DoErrorLog( x... ) { IOLog( "ERROR: " x ) ; }
#define DoDebugLog( x... ) { IOLog( x ) ; }

#define ErrorLog(x...) 				DoErrorLog( x ) ;
#define	ErrorLogCond( x, y... )		{ if (x) ErrorLog ( y ) ; }

#if IOFIREWIREDEBUG > 0
#	define DebugLogOrig(x...)			DoDebugLog( x ) ;
#	define DebugLogCondOrig( x, y... ) 	{ if (x) DebugLogOrig ( y ) ; }
#else
#	define DebugLogOrig(x...)			do {} while (0)
#	define DebugLogCondOrig( x, y... )	do {} while (0)
#endif

#define TIMEIT( doit, description ) \
{ \
	AbsoluteTime start, end; \
	IOFWGetAbsoluteTime( & start ); \
	{ \
		doit ;\
	}\
	IOFWGetAbsoluteTime( & end ); \
	SUB_ABSOLUTETIME( & end, & start ) ;\
	UInt64 nanos ;\
	absolutetime_to_nanoseconds( end, & nanos ) ;\
	DebugLogOrig("%s duration %llu us\n", "" description, nanos/1000) ;\
}

#define InfoLog(x...) do {} while (0)


