/*
botw-qt-backend
Copyright (C) 2023 iTNTPiston

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License along
with this program. If not, see <https://www.gnu.org/licenses/>
*/

#ifndef _BOTWQT_SETTINGS_H_
#define _BOTWQT_SETTINGS_H_

#include <stdint.h>

namespace botwqt {
// Internal settings struct
// The values here are not the defaults in UI. See botwqt_get_defaults for that.
struct Settings {
    // Should overlay the processed image on top of the frame
    bool enable_preview = false;
    // threshold to turn a pixel white
    uint32_t threshold = 0;
    // how many frames to wait before processing next
    uint32_t interval = 0;
    // port
    uint16_t port = 0;
};
}

#endif