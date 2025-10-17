/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 18, 2022.
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
#include "DALog.h"

#include "DABase.h"
#include "DAInternal.h"


#include <os/log.h>
#include <syslog.h>

static Boolean __gDALogDebug            = FALSE;
static FILE *  __gDALogDebugFile        = NULL;
static char *  __gDALogDebugHeaderLast  = NULL;
static char *  __gDALogDebugHeaderNext  = NULL;
static Boolean __gDALogDebugHeaderReset = FALSE;
static Boolean __gDALogError            = FALSE;
static os_log_t __gDALog                   = NULL;

static void __DALogInternalDebug( char * message )
{
    time_t clock = time( NULL );
    char   stamp[10];

    if ( strftime( stamp, sizeof( stamp ), "%T ", localtime( &clock ) ) )
    {
        fprintf( __gDALogDebugFile, "%s", stamp );
    }

    fprintf( __gDALogDebugFile, "%s", message );
    fprintf( __gDALogDebugFile, "\n" );
    fflush( __gDALogDebugFile );
}

static void __DALog( int level, const char * format, va_list arguments )
{
    char * message;

    if ( arguments )
    {
        message = ___CFStringCreateCStringWithFormatAndArguments( format, arguments );
    }
    else
    {
        message = strdup( format );
    }

    if ( message )
    {
        switch ( level )
        {
            case LOG_DEBUG:
            {
                if ( __gDALogDebug )
                {
                    if ( __gDALogDebugFile )
                    {
                        __DALogInternalDebug( message );
                    }
                }

                os_log_info(__gDALog ,"%{public}s" , message);

                break;
            }
            case LOG_ERR:
            {
                os_log_error(__gDALog, "%{public}s", message );

                break;
            }

            case LOG_INFO:
            {
                if ( __gDALogDebug )
                {
                    if ( __gDALogDebugFile )
                    {
                        __DALogInternalDebug( message );
                    }
                }
                os_log(__gDALog ,"%{public}s" , message);
                
                break;
            }

            default:
            {
                os_log(__gDALog ,"%{public}s" , message);

                break;
            }
        }

        free( message );
    }
}

void DALog( const char * format, ... )
{
    va_list arguments;

    va_start( arguments, format );

    __DALog( LOG_NOTICE, format, arguments );

    va_end( arguments );
}

  /* Remove in next version */
void DALogClose( void )
{
    __gDALogDebug   = FALSE;
    __gDALogError   = FALSE;


    if ( __gDALogDebugFile )
    {
        fclose( __gDALogDebugFile );

        __gDALogDebugFile = NULL;
    }

    closelog( );
}

void DALogDebug( const char * format, ... )
{
    va_list arguments;

    va_start( arguments, format );

    if ( __gDALogDebugHeaderReset )
    {
        assert( __gDALogDebugHeaderNext );

        if ( __gDALogDebugHeaderLast )
        {
            free( __gDALogDebugHeaderLast );
        }

        __gDALogDebugHeaderLast  = __gDALogDebugHeaderNext;
        __gDALogDebugHeaderNext  = NULL;
        __gDALogDebugHeaderReset = FALSE;

        __DALog( LOG_DEBUG, __gDALogDebugHeaderLast, NULL );
    }

    __DALog( LOG_DEBUG, format, arguments );

    va_end( arguments );
}

void DALogDebugHeader( const char * format, ... )
{
    va_list arguments;

    va_start( arguments, format );

    if ( __gDALogDebugHeaderNext )
    {
        free( __gDALogDebugHeaderNext );

        __gDALogDebugHeaderNext  = NULL;
        __gDALogDebugHeaderReset = FALSE;
    }

    if ( format )
    {
        char * header;

        header = ___CFStringCreateCStringWithFormatAndArguments( format, arguments );

        if ( header )
        {
            if ( __gDALogDebugHeaderLast )
            {
                if ( strcmp( __gDALogDebugHeaderLast, header ) )
                {
                    __gDALogDebugHeaderNext  = header;
                    __gDALogDebugHeaderReset = TRUE;
                }
                else
                {
                    free( header );
                }
            }
            else
            {
                __gDALogDebugHeaderNext  = header;
                __gDALogDebugHeaderReset = TRUE;
            }
        }
    }

    va_end( arguments );
}

void DALogError( const char * format, ... )
{
    va_list arguments;

    va_start( arguments, format );

    __DALog( LOG_DEBUG, format, arguments );

    va_end( arguments );

    va_start( arguments, format );

    __DALog( LOG_ERR, format, arguments );

    va_end( arguments );
}

void DALogInfo( const char * format, ... )
{
    va_list arguments;

    va_start( arguments, format );

    __DALog( LOG_INFO, format, arguments );

    va_end( arguments );
}

void DALogOpen( char * name, Boolean debug, Boolean error )
{

    __gDALog = os_log_create(_kDADaemonName, "default");
    /* Remove in next version */
    openlog( name, LOG_PID, LOG_DAEMON );

    if ( debug )
    {
        char * path;

        asprintf( &path, "/var/log/%s.log", name );

        if ( path )
        {
            __gDALogDebugFile = fopen( path, "a" );

            free( path );
        }
    }

    __gDALogDebug   = debug;
    __gDALogError   = error;

}


