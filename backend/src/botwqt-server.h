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

#ifndef _BOTWQT_SERVER_H_
#define _BOTWQT_SERVER_H_

#include <set>
#include <thread>
#include <mutex>
#include <QtCore/qthreadpool.h>
#include <asio.hpp>
#include <websocketpp/config/asio_no_tls.hpp>
#include <websocketpp/server.hpp>

// OpenCV Forward Declarations
namespace cv {
class Mat;
}

namespace botwqt {

class Frame;

class Server {
public:
    Server();
    ~Server();

    void start(uint16_t port);
    void stop();
    void send_frame(Frame& frame);
    
private:
    // underlying server
    websocketpp::server<websocketpp::config::asio> m_server;
    // server thread (handles connection)
    std::thread m_server_thread;
    // broadcast threads
    QThreadPool m_broadcast_thread_pool;

    bool m_frame_updated = false;
    // clients (access by both connection and broadcast threads)
    std::set<websocketpp::connection_hdl, std::owner_less<websocketpp::connection_hdl>> m_clients;
    // mutex
    std::mutex m_clients_mutex;

    void run_server();
    void on_open(websocketpp::connection_hdl hdl);
	void on_close(websocketpp::connection_hdl hdl);
    
};
}

#endif