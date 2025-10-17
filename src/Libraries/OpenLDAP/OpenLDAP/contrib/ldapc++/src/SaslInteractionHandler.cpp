/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 4, 2025.
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

// $OpenLDAP$
/*
 * Copyright 2007-2011 The OpenLDAP Foundation, All Rights Reserved.
 * COPYING RESTRICTIONS APPLY, see COPYRIGHT file
 */

#include <iostream>
#include <iomanip>
#include <limits>
#include "config.h"

#ifdef HAVE_TERMIOS_H
#include <termios.h>
#include <cstdio>
#endif

#include <string.h>
#include "SaslInteractionHandler.h"
#include "SaslInteraction.h"
#include "debug.h"

void DefaultSaslInteractionHandler::handleInteractions( 
        const std::list<SaslInteraction*> &cb ) 
{
    DEBUG(LDAP_DEBUG_TRACE, "DefaultSaslInteractionHandler::handleCallbacks()" 
            << std::endl );
    std::list<SaslInteraction*>::const_iterator i;

    for (i = cb.begin(); i != cb.end(); i++ ) {
        bool noecho;

        cleanupList.push_back(*i);

        std::cout << (*i)->getPrompt();
        if (! (*i)->getDefaultResult().empty() ) {
            std::cout << "(" << (*i)->getDefaultResult() << ")" ;
        }
        std:: cout << ": ";

        switch ( (*i)->getId() ) {
            case SASL_CB_PASS:
            case SASL_CB_ECHOPROMPT:
                noecho = true;
                noecho = true;
            break;
            default:
                noecho = false;
            break;
        }
#ifdef HAVE_TERMIOS_H
        /* turn off terminal echo if needed */
        struct termios old_attr;
        if ( noecho ) {
            struct termios attr;
            if (tcgetattr(STDIN_FILENO, &attr) < 0) {
                perror("tcgetattr");
            }

            /* save terminal attributes */
            memcpy(&old_attr, &attr, sizeof(attr));

            /* disable echo */
            attr.c_lflag &= ~(ECHO);

            /* write attributes to terminal */
            if (tcsetattr(STDIN_FILENO, TCSAFLUSH, &attr) < 0) {
                perror("tcsetattr");
            }
        }
#endif /* HAVE_TERMIOS_H */
        std::string input;
        std::cin >> std::noskipws >> input;
        std::cin >> std::skipws;
        (*i)->setResult(input);
        if( std::cin.fail() ) {
            std::cin.clear();
        }
        /* ignore the rest of the input line */
        std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

#ifdef HAVE_TERMIOS_H
        /* restore terminal settings */
        if ( noecho ) {
            tcsetattr(STDIN_FILENO, TCSANOW, &old_attr);
            std::cout << std::endl;
        }
#endif /* HAVE_TERMIOS_H */
    }
}

DefaultSaslInteractionHandler::~DefaultSaslInteractionHandler()
{
    DEBUG(LDAP_DEBUG_TRACE, "DefaultSaslInteractionHandler::~DefaultSaslInteractionHandler()"
            << std::endl );

    std::list<SaslInteraction*>::const_iterator i;
    for (i = cleanupList.begin(); i != cleanupList.end(); i++ ) {
        delete(*i);
    }
}
