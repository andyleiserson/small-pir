# Small PIR

Small PIR is a communication-efficient PIR based on the Ring Learning with
Errors (RLWE) problem. It is particularly compelling for databases of
modest size (hundreds of megabytes). For larger databases, the computation
required to answer a query (which is proportional to the database size)
grows, and other PIR schemes that transmit additional data to enable
more efficient computation become superior.

More details of the algorithm may be found in the [paper](./docs/paper.pdf).

This repository contains an implementation of Small PIR in Rust. It also
includes a [parameter search script](./tools/params.py).

This code has not been subject to review or audit, and any use is at your
own risk.
