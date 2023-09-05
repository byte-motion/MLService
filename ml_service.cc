#include "async_server.h"
#include "sync_server.h"

int main(int argc, char **argv)
{
    std::string port = argc > 1 ? std::string(argv[1]) : "50055";

    if (argc > 2 && argv[2] == "async")
    {
        byte_motion::ServerImpl server;
        server.Run(port);
    }
    else
    {
        byte_motion::MLServiceImpl service;
        service.Run(port);
    }

    return 0;
};
