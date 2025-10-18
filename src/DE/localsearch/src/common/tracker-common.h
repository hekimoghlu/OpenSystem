/*
 * Copyright (C) 2008, Nokia <ivan.frade@nokia.com>
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, write to the
 * Free Software Foundation, Inc., 51 Franklin Street, Fifth Floor,
 * Boston, MA  02110-1301, USA.
 */

#ifndef __LIBTRACKER_COMMON_H__
#define __LIBTRACKER_COMMON_H__

#include <glib.h>

#ifdef HAVE_POWER
#include "tracker-power.h"
#endif

#include "tracker-dbus.h"
#include "tracker-debug.h"
#include "tracker-enums.h"
#include "tracker-error-report.h"
#include "tracker-extract-info.h"
#include "tracker-file-utils.h"
#include "tracker-ioprio.h"

#ifdef HAVE_LANDLOCK
#include "tracker-landlock.h"
#endif

#include "tracker-miner.h"
#include "tracker-miner-proxy.h"
#include "tracker-sched.h"
#include "tracker-seccomp.h"
#include "tracker-systemd.h"
#include "tracker-term-utils.h"
#include "tracker-type-utils.h"
#include "tracker-utils.h"
#include "tracker-locale.h"
#include "tracker-miners-enum-types.h"
#include "tracker-module-manager.h"

#endif /* __LIBTRACKER_COMMON_H__ */
