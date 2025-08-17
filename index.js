import express, { urlencoded } from "express";
import cors from "cors";
import cookieParser from "cookie-parser";
import dotenv from "dotenv";
import connectDB from "./utils/db.js";
import userRoutes from "./routes/user.routes.js";
import groupRoutes from "./routes/group.routes.js";
import { app, server } from "./socket/socket.js";
// import path from "path";

dotenv.config({});

const PORT = process.env.PORT || 8000;

//middlewares
app.use(express.json());
app.use(cookieParser());
app.use(urlencoded({ extended: true }));

const corsOptions = {
  origin: process.env.URL || "http://localhost:5173",
  credentials: true,
};
app.use(cors(corsOptions));

app.use("/api/user", userRoutes);
app.use("/api/group", groupRoutes);

server.listen(PORT, () => {
  connectDB();
  console.log(`Server listen at port ${PORT}`);
});
