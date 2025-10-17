/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 30, 2024.
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
/* ACKNOWLEDGEMENTS:
 * This work was initially developed by Dmitry Kovalev for inclusion
 * by OpenLDAP Software.  Additional significant contributors include
 * Pierangelo Masarati.
 */

#include "portable.h"

#include <stdio.h>
#include <sys/types.h>
#include "ac/string.h"

#include "slap.h"
#include "proto-sql.h"

static backsql_api *backsqlapi;

int
backsql_api_config( backsql_info *bi, const char *name, int argc, char *argv[] )
{
	backsql_api	*ba;

	assert( bi != NULL );
	assert( name != NULL );

	for ( ba = backsqlapi; ba; ba = ba->ba_next ) {
		if ( strcasecmp( name, ba->ba_name ) == 0 ) {
			backsql_api	*ba2;

			ba2 = ch_malloc( sizeof( backsql_api ) );
			*ba2 = *ba;

			if ( ba2->ba_config ) {
				if ( ( *ba2->ba_config )( ba2, argc, argv ) ) {
					ch_free( ba2 );
					return 1;
				}
				ba2->ba_argc = argc;
				if ( argc ) {
					int i;
					ba2->ba_argv = ch_malloc( argc * sizeof(char *));
					for ( i=0; i<argc; i++ )
						ba2->ba_argv[i] = ch_strdup( argv[i] );
				}
			}
			
			ba2->ba_next = bi->sql_api;
			bi->sql_api = ba2;
			return 0;
		}
	}

	return 1;
}

int
backsql_api_destroy( backsql_info *bi )
{
	backsql_api	*ba;

	assert( bi != NULL );

	ba = bi->sql_api;

	if ( ba == NULL ) {
		return 0;
	}

	for ( ; ba; ba = ba->ba_next ) {
		if ( ba->ba_destroy ) {
			(void)( *ba->ba_destroy )( ba );
		}
	}

	return 0;
}

int
backsql_api_register( backsql_api *ba )
{
	backsql_api	*ba2;

	assert( ba != NULL );
	assert( ba->ba_private == NULL );

	if ( ba->ba_name == NULL ) {
		fprintf( stderr, "API module has no name\n" );
		exit(EXIT_FAILURE);
	}

	for ( ba2 = backsqlapi; ba2; ba2 = ba2->ba_next ) {
		if ( strcasecmp( ba->ba_name, ba2->ba_name ) == 0 ) {
			fprintf( stderr, "API module \"%s\" already defined\n", ba->ba_name );
			exit( EXIT_FAILURE );
		}
	}

	ba->ba_next = backsqlapi;
	backsqlapi = ba;

	return 0;
}

int
backsql_api_dn2odbc( Operation *op, SlapReply *rs, struct berval *dn )
{
	backsql_info	*bi = (backsql_info *)op->o_bd->be_private;
	backsql_api	*ba;
	int		rc;
	struct berval	bv;

	ba = bi->sql_api;

	if ( ba == NULL ) {
		return 0;
	}

	ber_dupbv( &bv, dn );

	for ( ; ba; ba = ba->ba_next ) {
		if ( ba->ba_dn2odbc ) {
			/*
			 * The dn2odbc() helper is supposed to rewrite
			 * the contents of bv, freeing the original value
			 * with ch_free() if required and replacing it 
			 * with a newly allocated one using ch_malloc() 
			 * or companion functions.
			 *
			 * NOTE: it is supposed to __always__ free
			 * the value of bv in case of error, and reset
			 * it with BER_BVZERO() .
			 */
			rc = ( *ba->ba_dn2odbc )( op, rs, &bv );

			if ( rc ) {
				/* in case of error, dn2odbc() must cleanup */
				assert( BER_BVISNULL( &bv ) );

				return rc;
			}
		}
	}

	assert( !BER_BVISNULL( &bv ) );

	*dn = bv;

	return 0;
}

int
backsql_api_odbc2dn( Operation *op, SlapReply *rs, struct berval *dn )
{
	backsql_info	*bi = (backsql_info *)op->o_bd->be_private;
	backsql_api	*ba;
	int		rc;
	struct berval	bv;

	ba = bi->sql_api;

	if ( ba == NULL ) {
		return 0;
	}

	ber_dupbv( &bv, dn );

	for ( ; ba; ba = ba->ba_next ) {
		if ( ba->ba_dn2odbc ) {
			rc = ( *ba->ba_odbc2dn )( op, rs, &bv );
			/*
			 * The odbc2dn() helper is supposed to rewrite
			 * the contents of bv, freeing the original value
			 * with ch_free() if required and replacing it 
			 * with a newly allocated one using ch_malloc() 
			 * or companion functions.
			 *
			 * NOTE: it is supposed to __always__ free
			 * the value of bv in case of error, and reset
			 * it with BER_BVZERO() .
			 */
			if ( rc ) {
				/* in case of error, odbc2dn() must cleanup */
				assert( BER_BVISNULL( &bv ) );

				return rc;
			}
		}
	}

	assert( !BER_BVISNULL( &bv ) );

	*dn = bv;

	return 0;
}

